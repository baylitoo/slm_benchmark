"""Classify a deployment failure into a coarse, display-friendly kind.

Pure and derived: the reconciler already publishes ``state`` + ``last_error``
for every deployment, and the supervisor now folds the killing signal / exit
code into ``last_error`` (see ``_exit_reason``). This maps those two strings to
one of a small set of failure kinds the UI can badge — no new database column,
no migration. ``None`` means "not a failure to surface".
"""

from __future__ import annotations

# A kind -> a short human label the UI can show next to the badge. The keys are
# the values classify_failure returns; the UI owns colour.
FAILURE_LABELS: dict[str, str] = {
    "oom": "Out of memory",
    "insufficient-memory": "Won't fit",
    "port-conflict": "Port in use",
    "crashed": "Crashed",
    "unhealthy": "Unhealthy",
    "spawn-error": "Won't start",
}

_FAILURE_STATES = {"failed", "degraded"}


def classify_failure(state: str | None, last_error: str | None) -> str | None:
    """A coarse failure kind for the UI, or ``None`` when not a failure.

    Order matters — the most specific, most useful diagnosis wins. A model that
    was OOM-killed AND then refused a restart carries both an OOM signal note
    and a fit-check reason in ``last_error``; ``oom`` (what actually happened)
    is reported over ``insufficient-memory`` (why it won't come back).
    """
    st = (state or "").strip().lower()
    err = (last_error or "").strip().lower()

    # A withheld restart / port collision is a failure even if the lifecycle
    # state momentarily reads otherwise, so key off the text too.
    is_failure = (
        st in _FAILURE_STATES
        or "restart withheld" in err
        or "collision" in err
        or "out of memory" in err
    )
    if not is_failure:
        return None

    if any(m in err for m in ("out of memory", "signal 9", "sigkill", "cannot allocate")):
        return "oom"
    if "restart withheld" in err and any(m in err for m in ("fit-check", "free", "margin")):
        return "insufficient-memory"
    if any(m in err for m in ("collision", "address already in use", "bind")):
        return "port-conflict"
    if "missing binary" in err or "failed to launch" in err or "not found" in err:
        return "spawn-error"
    if "exited with code" in err or "killed by" in err or "process exited" in err:
        return "crashed"
    if "health" in err or st == "degraded":
        return "unhealthy"
    return "crashed"
