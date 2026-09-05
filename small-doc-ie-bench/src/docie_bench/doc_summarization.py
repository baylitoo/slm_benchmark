"""Best-effort rolling document summarization (#430).

Fired once, fire-and-forget, right after a session document is saved (see
``studio_api.extract.upload_session_document``) -- never awaited by the
upload request itself, since a small model's summarization pass can take
longer than a caller should have to wait just to get a ``stored_name`` back.
docs-search's ``list_files`` reads whatever sidecar state exists at call
time, including "still summarizing", rather than the upload route blocking
until a summary lands.

Deliberately model-size-agnostic: the target profile is an operator setting
(``Settings.doc_summary_model``, e.g. ``"store:lfm2.5-350m"`` or
``"store:lfm2.5-2.6b"``), not a name baked in here, and resolves through the
same ``resolve_extraction_profile`` every chat/extract route already uses --
so pointing this at a bigger deployed model needs a config change, not a
code change. Unset (the default) makes ``summarize_document`` a no-op:
this is enrichment, not a hard dependency docs-search needs to function.

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

from docie_bench.mcp_servers.docs_search import (
    extract_page_texts,
    write_summary_state,
)
from docie_bench.serving.placement_resolver import (
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
    settings = get_settings()
    model_name = settings.doc_summary_model
    if not model_name:
        return
    write_summary_state(document_path, "summarizing")
    try:
        profile = resolve_extraction_profile(model_profile=model_name)
    except (PlacementNotFoundError, PlacementNotReadyError, ProfileResolutionError) as exc:
        logger.info("doc summarization skipped for %s: %s", document_path, exc)
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
