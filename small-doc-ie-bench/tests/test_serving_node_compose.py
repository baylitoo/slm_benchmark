"""The remote serving-node compose file (#538, milestone #4).

Nothing here starts a container. These are the invariants that cannot be caught
by reading the file once, because they break silently: a node that publishes a
compose service name as its address answers with the wrong model, a port window
that drifts from the published range records unreachable endpoints, and a
mounted worker volume would make the node look shareable when it is not.

Also pins the file against the central `serving` service, so the two cannot
drift apart unnoticed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_COMPOSE = REPO_ROOT / "docker-compose.serving-node.yml"
CENTRAL_COMPOSE = REPO_ROOT / "docker-compose.yml"
NODE_ENV_EXAMPLE = REPO_ROOT / ".env.serving-node.example"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def node() -> dict[str, Any]:
    return _load(NODE_COMPOSE)["services"]["serving"]


@pytest.fixture(scope="module")
def central() -> dict[str, Any]:
    return _load(CENTRAL_COMPOSE)["services"]["serving"]


def test_a_node_runs_the_serving_role_and_nothing_else(node: dict[str, Any]) -> None:
    services = _load(NODE_COMPOSE)["services"]

    assert list(services) == ["serving"]
    assert node["environment"]["DOCIE_WORKER_ROLE"] == "serving"


def test_a_node_depends_on_no_local_infrastructure(node: dict[str, Any]) -> None:
    """Postgres and Inngest live on the central host and are reached over the
    network, so a `depends_on` here could only name a service that does not
    exist on this machine."""
    assert "depends_on" not in node


@pytest.mark.parametrize(
    "variable",
    [
        "DOCIE_DB_HOST",
        "DOCIE_SERVING_ADVERTISE_HOST",
        "DOCIE_SERVING_PUBLISH_IP",
        "POSTGRES_PASSWORD",
    ],
)
def test_the_values_that_fail_silently_have_no_default(variable: str) -> None:
    """Each of these misroutes or over-exposes when guessed, so compose must
    refuse to start rather than substitute something plausible. ``:?`` is the
    required-variable form; ``:-`` would be a default."""
    raw = NODE_COMPOSE.read_text(encoding="utf-8")

    assert f"${{{variable}:?" in raw, f"{variable} must use the ${{VAR:?message}} form"
    assert f"${{{variable}:-" not in raw, f"{variable} must not carry a default"


def test_the_database_host_is_not_the_central_service_name(node: dict[str, Any]) -> None:
    """`postgres` resolves only inside the central compose network; a node
    pointed there reaches no database, or worse, its own."""
    url = node["environment"]["DATABASE_URL"]

    assert "@postgres:" not in url
    assert "${DOCIE_DB_HOST" in url


def test_the_advertise_host_is_never_defaulted_to_a_service_name() -> None:
    """The failure this guards is the expensive one: the central api resolves
    the recorded endpoint in its OWN network, so `serving` points back at the
    central container and a request for this node's model is answered by
    whatever holds the same port there — HTTP 200, wrong model."""
    raw = NODE_COMPOSE.read_text(encoding="utf-8")

    assert "DOCIE_SERVING_ADVERTISE_HOST: ${DOCIE_SERVING_ADVERTISE_HOST:?" in raw
    assert "DOCIE_SERVING_ADVERTISE_HOST:-serving" not in raw


def test_the_published_range_is_the_window_deploys_allocate_from(
    node: dict[str, Any],
) -> None:
    """A deploy landing on a port outside the published range records an
    endpoint nothing can reach. Both sides are driven by the same two
    variables, with the same defaults, so they cannot disagree."""
    environment = node["environment"]
    start = environment["DOCIE_SERVING_PORT_RANGE_START"]
    end = environment["DOCIE_SERVING_PORT_RANGE_END"]
    (published,) = node["ports"]

    # The first pick matches the code default, so a single-deploy node behaves
    # exactly like the central one.
    assert start.endswith(":-8088}")
    # Drift is what this really guards: the binding must interpolate the SAME
    # two expressions, on both the host and the container side.
    assert published.count(start) == 2
    assert published.count(end) == 2


def test_runtime_ports_are_bound_to_one_explicit_interface(node: dict[str, Any]) -> None:
    """A deployed runtime has no authentication of its own, so publishing the
    window on all interfaces of a remote host exposes free inference."""
    (published,) = node["ports"]

    assert published.startswith("${DOCIE_SERVING_PUBLISH_IP:?")
    # noqa S104: this asserts the all-interfaces bind is ABSENT.
    assert not published.startswith("0.0.0.0")  # noqa: S104
    assert not published.startswith("8088")  # bare host ports mean all interfaces


@pytest.mark.parametrize(
    "shared",
    ["/app/data", "/app/runs", "/app/annotations", "/app/artifacts"],
)
def test_no_worker_surface_is_mounted(node: dict[str, Any], shared: str) -> None:
    """These are benchmark/worker surfaces the serving role never writes.
    Mounting them would imply a node needs shared storage, which is what makes
    the fleet look harder to deploy than it is."""
    targets = [entry.split(":")[-2] for entry in node["volumes"] if entry.count(":") >= 2]
    targets += [
        entry.split(":")[1] for entry in node["volumes"] if entry.count(":") == 1
    ]

    assert shared not in targets


def test_the_node_state_volume_is_node_local(node: dict[str, Any]) -> None:
    """Desired state, the blob store and the node id all live here; two
    machines sharing it would fight over one deployments.json."""
    assert node["environment"]["DOCIE_SERVING_HOME"] == "/app/.serving"
    assert "serving-state:/app/.serving" in node["volumes"]
    assert "serving-state" in _load(NODE_COMPOSE)["volumes"]


# --- drift against the central serving service ---------------------------


def test_the_image_is_built_the_same_way_as_the_central_serving_service(
    node: dict[str, Any], central: dict[str, Any]
) -> None:
    """Encoder deployments spawn inside the container, so a node missing the
    extras would accept an encoder deploy and fail at launch."""
    assert node["build"]["dockerfile"] == central["build"]["dockerfile"]
    assert node["build"]["args"]["PIP_EXTRAS"] == central["build"]["args"]["PIP_EXTRAS"]
    assert node["build"]["args"]["INSTALL_EMBEDDINGS"] == (
        central["build"]["args"]["INSTALL_EMBEDDINGS"]
    )


@pytest.mark.parametrize(
    "key",
    ["DOCIE_WORKER_ROLE", "DOCIE_SERVING_HOME", "DOCIE_SERVING_BIND_HOST", "DATABASE_URL"],
)
def test_every_env_key_the_central_serving_service_needs_is_present(
    node: dict[str, Any], central: dict[str, Any], key: str
) -> None:
    """A key added to the central serving service and forgotten here is the
    likely drift, and it would only show up on a node."""
    assert key in central["environment"]
    assert key in node["environment"]


def test_the_node_runs_the_same_entrypoint(
    node: dict[str, Any], central: dict[str, Any]
) -> None:
    assert node["command"] == central["command"]


# --- the example env ------------------------------------------------------


@pytest.mark.parametrize(
    "variable",
    [
        "DOCIE_DB_HOST",
        "DOCIE_SERVING_ADVERTISE_HOST",
        "DOCIE_SERVING_PUBLISH_IP",
        "DOCIE_SERVING_NODE_ID",
        "INNGEST_BASE_URL",
        "INNGEST_CONNECT_GATEWAY_URL",
    ],
)
def test_the_example_env_covers_every_per_node_value(variable: str) -> None:
    assert f"{variable}=" in NODE_ENV_EXAMPLE.read_text(encoding="utf-8")


def test_the_example_env_does_not_ship_a_usable_advertise_host() -> None:
    """An operator who pastes the example unedited must get a refusal, not a
    node that silently advertises something wrong."""
    for line in NODE_ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if line.startswith("DOCIE_SERVING_ADVERTISE_HOST="):
            assert line.split("=", 1)[1].strip() == ""
            return
    pytest.fail("DOCIE_SERVING_ADVERTISE_HOST is not in the example env")


# --- the central side a node has to reach --------------------------------


@pytest.mark.parametrize(
    ("service", "variable"),
    [("postgres", "POSTGRES_BIND_HOST"), ("inngest", "INNGEST_BIND_HOST")],
)
def test_the_central_services_a_node_needs_have_a_binding_knob(
    service: str, variable: str
) -> None:
    """A node reaches Postgres and Inngest over the private network, so both
    have to be bindable without editing the tracked compose file."""
    published = _load(CENTRAL_COMPOSE)["services"][service]["ports"]

    assert published, f"{service} publishes no host port"
    assert all(str(entry).startswith(f"${{{variable}:-") for entry in published)


@pytest.mark.parametrize(
    ("service", "variable"),
    [("postgres", "POSTGRES_BIND_HOST"), ("inngest", "INNGEST_BIND_HOST")],
)
def test_the_central_services_stay_on_loopback_by_default(
    service: str, variable: str
) -> None:
    """Opening them is a fleet decision an operator makes explicitly: Postgres
    answers to real credentials and the Inngest dashboard has no auth at all,
    so a single-host stack must not expose either by default."""
    published = _load(CENTRAL_COMPOSE)["services"][service]["ports"]

    for entry in published:
        assert str(entry).startswith(f"${{{variable}:-127.0.0.1}}")


def test_the_node_example_says_the_central_side_must_be_opened_first() -> None:
    """The node values are useless until the central host binds these, and
    that is the step most likely to be missed."""
    text = NODE_ENV_EXAMPLE.read_text(encoding="utf-8")

    assert "POSTGRES_BIND_HOST" in text
    assert "INNGEST_BIND_HOST" in text

# --- the cache-root upgrade (#595 2a) -------------------------------------
#
# A node that already holds weights keeps them at $DOCIE_SERVING_HOME/models,
# inside the serving-state volume. Adding a volume for the separated cache
# root is safe only while nothing mounts over that path: an empty volume
# mounted there would hide every seeded model, and the node would report an
# empty blob store and re-seed from scratch.
#
# These cannot prove Docker's mount semantics -- only a live upgrade does
# that. They pin the structural property that makes the upgrade safe, so it
# cannot regress unnoticed in a later edit.

WEIGHTS_PATH = "/app/.serving/models"


def _mount_targets(node: dict[str, Any]) -> list[str]:
    """Container-side path of every volume entry (``source:target[:opts]``)."""
    targets = []
    for entry in node.get("volumes", []):
        parts = str(entry).split(":")
        # A Windows-style or variable source can itself contain a colon, so the
        # target is the second-to-last part when options are present, else the
        # last. Both forms here are simple, and this keeps it honest.
        targets.append(parts[1] if len(parts) >= 2 else parts[0])
    return targets


def test_nothing_is_mounted_over_an_existing_nodes_weights(
    node: dict[str, Any],
) -> None:
    """The upgrade invariant. The weights live inside the serving-state volume;
    any OTHER mount at or above that path would shadow them."""
    shadowing = [
        target
        for target in _mount_targets(node)
        if target != "/app/.serving"
        and (target == WEIGHTS_PATH or WEIGHTS_PATH.startswith(target.rstrip("/") + "/"))
    ]

    assert shadowing == [], f"these mounts would hide seeded weights: {shadowing}"


def test_the_weights_volume_is_still_the_state_volume_by_default(
    node: dict[str, Any],
) -> None:
    """The compatibility default: with DOCIE_MODEL_STORE_DIR unset the weights
    are a child of the state root, so serving-state must still be mounted at
    exactly that root."""
    assert "/app/.serving" in _mount_targets(node)


def test_the_separate_cache_volume_is_not_mounted_by_default(
    node: dict[str, Any],
) -> None:
    """Opt-in on purpose. An always-mounted empty volume invites setting
    DOCIE_MODEL_STORE_DIR without moving the weights across, which makes the
    node report an empty store -- the migration is a documented procedure, not
    a one-line flip."""
    mounted = [entry for entry in node.get("volumes", []) if "model-cache" in str(entry)]

    assert mounted == [], f"model-cache should be opt-in, found: {mounted}"


def test_the_separate_cache_volume_is_declared_for_when_it_is_wanted() -> None:
    """Declared but unmounted, so enabling the split is one uncomment rather
    than two edits in two places."""
    assert "model-cache" in _load(NODE_COMPOSE).get("volumes", {})


def test_the_migration_is_documented_where_the_variable_is(
    ) -> None:
    """Setting the variable alone loses sight of the weights, so the env
    example must not present it as a single-line change."""
    text = NODE_ENV_EXAMPLE.read_text(encoding="utf-8")

    assert "DOCIE_MODEL_STORE_DIR" in text
    assert "fleet-runbook.md" in text, "the migration procedure must be pointed at"
