"""Every Inngest function is registered by exactly one role.

The failure this guards against is silent and total: a function can be defined,
decorated, triggered and tested, and still never run in a deployed system,
because registration happens in a list a thousand lines away from the
decorator. The event is accepted, the send succeeds, and nothing handles it.

That is what happened to ``batch_extract_job`` -- defined with a trigger,
covered by tests that called its handler directly, and in neither role list,
so batch extraction returned success and did nothing.

This is the Inngest-side equivalent of the Studio route-inventory test: a
router built but never mounted compiles fine and serves 404 for a whole
domain.
"""

from __future__ import annotations

import inspect

import inngest
import pytest

from docie_bench.inngest import functions as fn
from docie_bench.inngest.client import inngest_client, serving_client


def _declared_functions() -> dict[str, inngest.Function]:
    """Every Inngest function object defined in the functions module."""
    return {
        name: value
        for name, value in vars(fn).items()
        if isinstance(value, inngest.Function) and not name.startswith("_")
    }


def _registered() -> dict[str, list[str]]:
    """Function object id -> the role lists it appears in."""
    lists = {"worker": fn.worker_functions, "serving": fn.serving_functions}
    seen: dict[str, list[str]] = {}
    for role, registered in lists.items():
        for function in registered:
            seen.setdefault(str(id(function)), []).append(role)
    return seen


def test_every_declared_function_is_registered() -> None:
    declared = _declared_functions()
    registered = _registered()

    unregistered = sorted(
        name for name, function in declared.items() if str(id(function)) not in registered
    )

    # A function here is decorated, triggered, and unreachable. Add it to
    # worker_functions (scaled work) or serving_functions (single-writer,
    # spawns runtimes) -- whichever matches what it does.
    assert unregistered == [], (
        f"declared but registered by no role: {', '.join(unregistered)}. "
        "Its trigger event would be accepted and silently dropped."
    )


def test_no_function_is_registered_twice() -> None:
    declared = _declared_functions()
    registered = _registered()
    by_id = {str(id(function)): name for name, function in declared.items()}

    duplicated = sorted(
        f"{by_id.get(key, key)} in {'+'.join(roles)}"
        for key, roles in registered.items()
        if len(roles) > 1
    )

    # Both roles registering one function means two fleets claim it; the `all`
    # role already registers both app ids, so this is never what you want.
    assert duplicated == []


def test_batch_extraction_is_reachable() -> None:
    # The specific regression. Without the fix this fails, which is the whole
    # point of writing it as its own case rather than trusting the sweep above
    # to stay readable.
    assert fn.batch_extract_job in fn.worker_functions


@pytest.mark.parametrize("role", ["worker", "serving", "all"])
def test_each_role_resolves_to_a_non_empty_app_set(role: str) -> None:
    apps = fn.apps_for_role(role)

    assert apps, f"role {role!r} registers no apps"
    for client, registered in apps:
        assert registered, f"role {role!r} registers {client.app_id} with no functions"


def test_a_function_is_registered_on_the_client_it_was_created_with() -> None:
    """The invariant ``apps_for_role`` documents, enforced.

    A Connect handshake sends an app's function list as the authoritative set
    for that app id. A function created on one client but registered under the
    other would be synced to an app that never triggers it, and the two fleets
    would overwrite each other's lists on every reconnect.
    """
    expected = {
        inngest_client.app_id: fn.worker_functions,
        serving_client.app_id: fn.serving_functions,
    }

    for client, registered in fn.apps_for_role("all"):
        assert registered == expected[client.app_id], (
            f"app {client.app_id} is registered with the wrong function list"
        )


def test_unknown_roles_are_rejected_rather_than_silently_empty() -> None:
    # A typo in DOCIE_WORKER_ROLE must fail the process, not start a worker
    # that registers nothing and looks healthy.
    for bad in ("workers", "Serving ", "", "all-of-them"):
        if bad.strip().lower() in {"worker", "serving", "all", ""}:
            continue
        with pytest.raises(ValueError, match="DOCIE_WORKER_ROLE"):
            fn.apps_for_role(bad)
        with pytest.raises(ValueError, match="DOCIE_WORKER_ROLE"):
            fn.functions_for_role(bad)


def test_the_sweep_actually_sees_the_functions() -> None:
    # A guard that silently matches nothing passes forever. Pin that the
    # discovery finds the known jobs, so a refactor that changes how functions
    # are declared fails here instead of disabling the check.
    declared = _declared_functions()

    assert len(declared) >= 10
    for name in ("extract_document", "batch_extract_job", "run_benchmark_job"):
        assert name in declared, f"{name} not discovered -- the sweep is broken"
    assert inspect.ismodule(fn)
