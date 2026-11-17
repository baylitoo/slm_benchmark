"""Per-deployment recency sidecars: ``<serving-home>/recency/<name>``.

``last_served`` must be written by the code path that actually serves traffic
— the scaled ``worker`` extract path — but ``deployments.json`` has exactly ONE
writer (the single-replica ``serving`` service, P1). Resolution (design doc §1,
fix #5): each extract writes a tiny single-key sidecar file per deployment
(atomic ``os.replace``), and the serving-side reconciler folds the sidecars
into ``DeploymentRecord.last_served`` each cycle. ``last_served`` is a
monotonic max-timestamp, so last-write-wins between concurrent workers is the
CORRECT semantics and a lost update is harmless (an idle unload slips one
cycle at worst).

Everything here is best-effort: recency is an optimization input (LRU/idle in
PR-4), so a disk hiccup must never fail an extraction or a reconcile cycle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

from docie_bench.serving.paths import serving_home

logger = logging.getLogger(__name__)

# The catalog stamp is throttled per name: the idle TTL is minutes, and a
# benchmark stamps once per prediction.
CATALOG_STAMP_INTERVAL_S = 30.0
_catalog_stamped: dict[str, float] = {}

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def recency_dir(home: Path | None = None) -> Path:
    return (home if home is not None else serving_home()) / "recency"


def _filename(name: str) -> str:
    """A filesystem-safe sidecar filename for a deployment name.

    Deployment names are aliases and normally already safe; anything else
    (path separators, ``:`` on Windows, ...) is mapped to a stable digest so a
    hostile or odd name can never traverse out of the recency dir. Both the
    stamper and the folder use this same mapping, so they always agree.
    """
    if _SAFE_NAME.match(name):
        return name
    return "sha256-" + hashlib.sha256(name.encode("utf-8")).hexdigest()


def stamp(name: str, *, timestamp: float | None = None, home: Path | None = None) -> None:
    """Record "deployment ``name`` served a request now" (best-effort, atomic).

    Single-key write via temp-file + ``os.replace``: concurrent stampers
    last-write-wins, which is correct for a monotonic max-timestamp. Never
    raises — recency must not fail the extraction that produced it.
    """
    try:
        directory = recency_dir(home)
        directory.mkdir(parents=True, exist_ok=True)
        value = timestamp if timestamp is not None else time.time()
        target = directory / _filename(name)
        temporary = directory / f".{target.name}.tmp"
        temporary.write_text(f"{value:.6f}", encoding="utf-8")
        os.replace(temporary, target)
    except OSError:
        logger.warning("could not stamp recency for deployment %r", name, exc_info=True)


def _deployment_record_names(home: Path | None = None) -> frozenset[str]:
    """Names of the deployment records in the shared ``deployments.json``.

    Best-effort direct read (no supervisor construction, no mkdir side
    effects): any hiccup reads as "no deployments", so a recency stamp can
    never fail the extraction that produced it.
    """
    path = (home if home is not None else serving_home()) / "deployments.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frozenset()
    deployments = payload.get("deployments") if isinstance(payload, dict) else None
    if not isinstance(deployments, dict):
        return frozenset()
    return frozenset(str(name) for name in deployments)


def stamp_served_profile(
    profile_name: str | None,
    *,
    deployment: str | None = None,
    timestamp: float | None = None,
    home: Path | None = None,
) -> None:
    """Stamp recency for whichever DEPLOYMENT served an extraction (best-effort).

    ``last_served`` is the PR-4 LRU/idle-TTL input, and it must be written by
    EVERY surface that serves traffic — the Inngest worker extract path, the
    direct API extract endpoints, and the benchmark runner — or deployments
    driven by the unstamped surfaces read as idle forever and become the first
    idle-TTL/eviction victims mid-use.

    Resolution order:

    * an explicit ``deployment`` selector is stamped verbatim (it IS the
      record name);
    * else ``profile_name`` is considered: a ``store:<name>`` profile (built
      by ``resolve_store_profile``) stamps the deployment record ``<name>``
      that the store deploy created (``serve_store_model`` names the record
      after the store entry), and a bare name is stamped iff it names a
      deployment record. A plain models.yaml profile is not a deployment and
      gets no sidecar.

    Never raises; never touches ``deployments.json`` (single-writer, P1).
    """
    from docie_bench.serving.placement_resolver import STORE_PROFILE_PREFIX

    target = str(deployment) if deployment else None
    if target is None and profile_name:
        candidate = str(profile_name)
        if candidate.startswith(STORE_PROFILE_PREFIX):
            candidate = candidate[len(STORE_PROFILE_PREFIX) :]
        if candidate in _deployment_record_names(home):
            target = candidate
    if target:
        stamp(target, timestamp=timestamp, home=home)
    # The local file only covers deployments this machine owns. The catalog
    # stamp reaches the owning node wherever it runs, so it is not gated on
    # the local deployments.json.
    catalog_name = target
    if catalog_name is None and profile_name:
        catalog_name = str(profile_name).removeprefix(STORE_PROFILE_PREFIX)
    if catalog_name:
        stamp_catalog(catalog_name, timestamp=timestamp)


def stamp_catalog(name: str, *, timestamp: float | None = None) -> None:
    """Stamp ``last_served_at`` on the placement row(s) for ``name``
    (best-effort, throttled, never raises)."""
    import datetime as dt

    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    now = time.monotonic()
    last = _catalog_stamped.get(name)
    if timestamp is None and last is not None and now - last < CATALOG_STAMP_INTERVAL_S:
        return
    _catalog_stamped[name] = now
    value = timestamp if timestamp is not None else time.time()
    try:
        ModelCatalog().touch_served(name, dt.datetime.fromtimestamp(value, dt.UTC))
    except CatalogUnavailableError:
        pass
    except Exception:  # noqa: BLE001 - recency must not fail the request
        logger.debug("catalog recency stamp failed for %r", name, exc_info=True)


def read_for(names: list[str], *, home: Path | None = None) -> dict[str, float]:
    """Read the sidecar timestamps for ``names`` (missing/corrupt => absent).

    Keyed by deployment name (the reconciler folds by record name); reads only
    the sidecars for known records, so stray files are simply ignored.
    """
    directory = recency_dir(home)
    result: dict[str, float] = {}
    for name in names:
        path = directory / _filename(name)
        try:
            result[name] = float(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
    return result


__all__ = ["stamp", "stamp_served_profile", "read_for", "recency_dir"]
