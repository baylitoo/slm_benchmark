"""Best-effort rolling document summarization (#430).

Fired once, fire-and-forget, right after a session document is saved (see
``studio_api.extract.upload_session_document``) -- never awaited by the
upload request itself, since a small model's summarization pass can take
longer than a caller should have to wait just to get a ``stored_name`` back.
docs-search's ``list_files`` reads whatever sidecar state exists at call
time, including "still summarizing", rather than the upload route blocking
until a summary lands.

Deliberately model-size-agnostic: the target profile is an operator setting
(``Settings.doc_summary_model``, default ``"store:lfm2.5-350m"``), not a
name baked in here, and resolves through the same ``resolve_extraction_profile``
every chat/extract route already uses -- so pointing this at a bigger
deployed model (e.g. ``"store:lfm2.5-2.6b"``) needs a config change, not a
code change. Empty/unset makes ``summarize_document`` a no-op: this is
enrichment, not a hard dependency docs-search needs to function.

The default profile is a ``store:`` model, which is very likely NOT already
live the first time an upload fires this (nothing else may have deployed
it yet). Mirrors ``chat_api._resolve_or_error``'s own load-on-demand seam:
on ``PlacementNotFoundError``/``PlacementNotReadyError``, fires
``trigger_deployment_load`` (same event a first chat request against that
model would fire) and then polls ``resolve_extraction_profile`` until it
comes up or a bounded timeout elapses -- safe here specifically because
this runs detached from any request a user is actually waiting on, unlike
the HTTP route this pattern is borrowed from.

The summary is built ROLLING, chunk_pages pages at a time (default 4, see
``Settings.doc_summary_chunk_pages``) -- each call folds the running summary
so far plus the next chunk into an updated summary capped at
``doc_summary_max_chars`` -- rather than one call over the whole document,
so a 200-page document costs the same per-call prompt size as a 10-page one
and the final result stays short regardless of document length.

Sidecar read/write (``read_summary``/``write_summary_state``/
``summary_sidecar_path``) lives in ``docs_search.py``, not here -- see that
module's comment above them for why: ``list_files`` needs to read a sidecar
cheaply from docs-search's own lightweight, fresh-subprocess-per-request
runtime, without pulling in the serving-stack imports this module needs
just to GENERATE one.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException

from docie_bench.inngest.serving_api import trigger_deployment_load
from docie_bench.mcp_servers.docs_search import (
    extract_page_texts,
    write_summary_state,
)
from docie_bench.serving.catalog import CatalogUnavailableError
from docie_bench.serving.placement_resolver import (
    STORE_PROFILE_PREFIX,
    PlacementNotFoundError,
    PlacementNotReadyError,
)
from docie_bench.serving.profile_resolver import (
    ProfileResolutionError,
    resolve_extraction_profile,
)
from docie_bench.settings import get_settings

logger = logging.getLogger(__name__)

_SUMMARY_MAX_TOKENS = 300
# Bounds on the load-on-demand poll below -- never wait longer than this
# regardless of what trigger_deployment_load's own ETA estimate says, so a
# stuck/failed deploy doesn't strand this background task indefinitely.
_MAX_LOAD_WAIT_SECONDS = 300.0
_LOAD_POLL_INTERVAL_SECONDS = 3.0

# asyncio only holds a WEAK reference to a task it didn't create via
# ensure_future-with-a-kept-handle -- an unreferenced task can be garbage
# collected mid-run. spawn_summarize_document keeps one here until the task
# finishes, discarding it via the done callback so this set doesn't grow
# unbounded across many uploads.
_background_tasks: set[asyncio.Task[None]] = set()


def spawn_summarize_document(document_path: Path) -> None:
    """Fire ``summarize_document`` detached from the current request."""
    task = asyncio.create_task(summarize_document(document_path))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _summarize_chunk(
    client: httpx.AsyncClient,
    profile: Any,
    running_summary: str | None,
    chunk_text: str,
    max_chars: int,
) -> str:
    prior = f'Summary so far: "{running_summary}"\n\n' if running_summary else ""
    prompt = (
        f"{prior}Update the summary using this next excerpt from the same document "
        f"(what it is, its subject, key parties/figures -- not a restatement of "
        f"every sentence). Keep the whole answer under {max_chars} characters, plain "
        f"text, no preamble.\n\nExcerpt:\n{chunk_text}"
    )
    body = {
        "model": profile.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": _SUMMARY_MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {profile.api_key}", "Content-Type": "application/json"}
    response = await client.post(
        f"{profile.base_url}/chat/completions", json=body, headers=headers
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("summarizer returned a non-text completion")
    return content.strip()[:max_chars]


async def _resolve_loading_on_demand(model_name: str, document_path: Path) -> Any | None:
    """``resolve_extraction_profile``, firing the deployment's own
    load-on-demand event and polling if it isn't live yet.

    Returns ``None`` if the model was never a real catalog entry (nothing to
    trigger), or if it didn't come up within ``_MAX_LOAD_WAIT_SECONDS`` --
    either way the caller writes an appropriate sidecar state itself.
    """
    try:
        return resolve_extraction_profile(model_profile=model_name)
    except (ProfileResolutionError, CatalogUnavailableError) as exc:
        logger.info("doc summarization: %r is not routable: %s", model_name, exc)
        return None
    except (PlacementNotFoundError, PlacementNotReadyError) as exc:
        logger.info("doc summarization: %s not live yet for %s: %s", model_name, document_path, exc)

    store_name = (
        model_name[len(STORE_PROFILE_PREFIX) :]
        if model_name.startswith(STORE_PROFILE_PREFIX)
        else None
    )
    triggered = None
    if store_name:
        try:
            triggered = await trigger_deployment_load(store_name)
        except HTTPException as exc:
            # trigger_deployment_load's "never deployed yet" branch fires the
            # deploy event via send_or_503, unguarded -- a send failure there
            # raises straight through rather than returning None like every
            # other "nothing to trigger" case this function already handles.
            logger.info("doc summarization: failed to trigger a load for %s: %s", model_name, exc)
    if triggered is None:
        return None
    _, eta_seconds = triggered
    deadline = min(eta_seconds * 2, _MAX_LOAD_WAIT_SECONDS)
    elapsed = 0.0
    while elapsed < deadline:
        await asyncio.sleep(_LOAD_POLL_INTERVAL_SECONDS)
        elapsed += _LOAD_POLL_INTERVAL_SECONDS
        try:
            return resolve_extraction_profile(model_profile=model_name)
        except (
            PlacementNotFoundError,
            PlacementNotReadyError,
            ProfileResolutionError,
            CatalogUnavailableError,
        ):
            continue
    logger.info(
        "doc summarization: %s never became ready within %.0fs for %s",
        model_name,
        deadline,
        document_path,
    )
    return None


async def summarize_document(
    document_path: Path, *, http_client: httpx.AsyncClient | None = None
) -> None:
    """Rolling-summarize ``document_path`` and write the result to its
    sidecar. Never raises -- this runs detached from whatever request
    triggered it (see module docstring), so there is no caller left to
    handle an exception; every failure mode instead lands in the sidecar's
    ``state`` for ``list_files`` to report honestly.

    ``http_client``, when given (tests only -- production always resolves
    its own, sized to the resolved profile's own timeout, same convention as
    ``agents.guard.guard_analyze``'s injected client), is reused across
    every chunk's call rather than opening a fresh connection pool per
    chunk of the same document.
    """
    try:
        await _summarize_document_impl(document_path, http_client=http_client)
    except Exception:
        # Last-resort backstop for the "never raises" promise above: every
        # anticipated failure is already handled inside the impl (a written
        # sidecar state per case), but an unanticipated one must still not
        # escape as an unhandled exception on a detached task -- and must
        # not leave the sidecar stuck on "summarizing" forever, which is
        # what list_files would otherwise show indefinitely.
        logger.exception("doc summarization: unexpected failure for %s", document_path)
        write_summary_state(document_path, "failed")


async def _summarize_document_impl(
    document_path: Path, *, http_client: httpx.AsyncClient | None = None
) -> None:
    settings = get_settings()
    model_name = settings.doc_summary_model
    if not model_name:
        return
    write_summary_state(document_path, "summarizing")
    profile = await _resolve_loading_on_demand(model_name, document_path)
    if profile is None:
        write_summary_state(document_path, "unavailable")
        return

    try:
        # extract_page_texts (liteparse + OCR fallback) is seconds-to-minutes
        # of blocking work on a scanned PDF -- this task runs on the SAME
        # event loop serving every other request, so it must not run this
        # inline (same hazard studio_api.extract.render_document's rasterize
        # already offloads via asyncio.to_thread).
        page_texts = await asyncio.to_thread(extract_page_texts, document_path)
    except Exception:
        logger.exception("doc summarization: text extraction failed for %s", document_path)
        write_summary_state(document_path, "failed")
        return
    if not page_texts:
        write_summary_state(document_path, "failed")
        return

    chunk_pages = settings.doc_summary_chunk_pages
    max_chars = settings.doc_summary_max_chars
    pages = sorted(page_texts)
    running_summary: str | None = None
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=profile.timeout_seconds)
    try:
        for start in range(0, len(pages), chunk_pages):
            chunk_numbers = pages[start : start + chunk_pages]
            chunk_text = "\n\n".join(page_texts[p] for p in chunk_numbers)
            running_summary = await _summarize_chunk(
                client, profile, running_summary, chunk_text, max_chars
            )
    except (httpx.HTTPError, LookupError, TypeError, ValueError) as exc:
        logger.info("doc summarization failed partway for %s: %s", document_path, exc)
        write_summary_state(document_path, "failed", running_summary)
        return
    finally:
        if owns_client:
            await client.aclose()
    write_summary_state(document_path, "ready", running_summary)
