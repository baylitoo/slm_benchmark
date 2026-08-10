"""classify_failure: map (state, last_error) to a coarse failure kind for the UI."""

from __future__ import annotations

import pytest

from docie_bench.serving.failure import FAILURE_LABELS, classify_failure


@pytest.mark.parametrize(
    ("state", "last_error", "expected"),
    [
        # OOM by signal — the supervisor's _exit_reason wording.
        (
            "failed",
            "process killed by SIGKILL (signal 9) — likely out of memory (OOM)",
            "oom",
        ),
        # OOM-killed AND then refused a restart: OOM (what happened) wins over
        # insufficient-memory (why it won't return).
        (
            "failed",
            "process killed by SIGKILL (signal 9) — likely out of memory (OOM) "
            "| restart withheld: fit-check failed: needs 5096407040 but only 3GB free",
            "oom",
        ),
        # Pure fit-check denial (too big to ever load) — never actually ran.
        (
            "failed",
            "runtime process exited | restart withheld: fit-check failed: "
            "needs 5096407040 bytes but only 3260660941 free minus the margin",
            "insufficient-memory",
        ),
        ("failed", "error: bind(): Address already in use", "port-conflict"),
        ("failed", "Failed to launch llamacpp: [Errno 2] No such file", "spawn-error"),
        ("failed", "process exited with code 1", "crashed"),
        ("degraded", "health check returned status 503", "unhealthy"),
        # Not a failure — nothing to badge.
        ("ready", None, None),
        ("cold", "", None),
        ("hot", None, None),
    ],
)
def test_classify_failure(state: str, last_error: str | None, expected: str | None) -> None:
    assert classify_failure(state, last_error) == expected


def test_every_kind_has_a_label() -> None:
    kinds = {"oom", "insufficient-memory", "port-conflict", "spawn-error", "crashed", "unhealthy"}
    assert kinds <= set(FAILURE_LABELS)
