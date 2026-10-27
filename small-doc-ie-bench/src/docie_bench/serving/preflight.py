"""Is this machine correctly configured to serve as a fleet node? (#538)

Every misconfiguration a new serving node can have is already documented
somewhere in this package, usually with a note saying it "fails as a wrong
answer, not as an error". That is the problem this module exists for. A node
with the wrong ``DOCIE_SERVING_ADVERTISE_HOST`` starts cleanly, publishes
capacity, accepts deploys, and then answers requests for its models from
whatever happens to hold the same port on another machine -- HTTP 200, wrong
model. A node pointed at the wrong database holds its working set and logs a
convergence warning nobody is reading. A node without
``DOCIE_SERVING_DESIRED_APPLY`` joins the fleet and silently ignores every
targeted deploy.

None of those are detectable by looking at the node, which is why bringing up
the second machine is the risky step and why it gets a preflight rather than a
runbook paragraph.

Two halves, deliberately separated:

* :func:`evaluate` is PURE -- facts in, verdicts out. Every check is a row in a
  table test, with no node, no database and no socket.
* :func:`collect_facts` is the impure half. Each probe is contained, so an
  unreachable database still lets the other checks report: a doctor that
  refuses to say anything because one probe failed is the opposite of useful.

Severity is about consequence, not about certainty. FAIL means this node cannot
do its job. WARN means it will appear to work and be wrong, or will silently do
less than asked -- the category most of these failures fall into. INFO is a
fact worth seeing that is nobody's fault.
"""

from __future__ import annotations

import datetime as dt
import enum
import logging
import os
import socket
from collections.abc import Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: Loopback and wildcard addresses that cannot work as an advertise host: the
#: central api resolves the recorded endpoint in ITS network, where "localhost"
#: is the api's own container.
UNROUTABLE_ADVERTISE_HOSTS = frozenset(
    # noqa below: these addresses are REJECTED here, never bound.
    {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}  # noqa: S104
)


class Severity(enum.Enum):
    PASS = "pass"  # noqa: S105 - a verdict name, not a credential
    INFO = "info"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class Check:
    """One verdict, and what it costs if ignored."""

    name: str
    severity: Severity
    detail: str
    #: What goes wrong in production when this is not PASS. Empty for PASS.
    consequence: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "detail": self.detail,
            "consequence": self.consequence or None,
        }


@dataclass(frozen=True)
class NodeFacts:
    """Everything :func:`evaluate` judges, gathered by :func:`collect_facts`.

    Optional fields are three-state on purpose: ``None`` means the probe could
    not run, which is not the same as a negative result and must not be
    reported as one.
    """

    worker_role: str = "all"
    serving_home: str = ""
    serving_home_writable: bool = True
    node_id: str | None = None
    #: Where the id came from: "env", "persisted", "generated", or "" when
    #: unknown. An operator needs to know which, because a generated id means
    #: THIS call minted it -- the doctor persists one on a fresh node, exactly
    #: as the reconciler's first publish would.
    node_id_source: str = ""
    advertise_host: str = ""
    #: Addresses ``advertise_host`` resolves to here; None when it does not
    #: resolve at all.
    advertise_resolved: tuple[str, ...] | None = None
    #: This host's own addresses, for comparison against the above.
    local_addresses: frozenset[str] = field(default_factory=frozenset)
    bind_host: str = "0.0.0.0"  # noqa: S104 - container service bind default
    port_range: tuple[int, int] = (8088, 8188)
    #: Ports in the window already held by something else; None when not probed.
    ports_in_use: tuple[int, ...] | None = None
    desired_apply: bool = False
    db_reachable: bool = False
    db_detail: str | None = None
    #: Seconds the database clock is ahead of this node's; None when unknown.
    db_clock_skew_s: float | None = None
    inngest_base_url: str | None = None
    inngest_reachable: bool | None = None
    #: Store models the catalog knows; None when the catalog could not be read.
    catalog_models: tuple[str, ...] | None = None
    #: Store models whose blob is on THIS node's disk.
    store_models: tuple[str, ...] | None = None
    #: Other nodes with a row in the catalog.
    other_node_ids: tuple[str, ...] | None = None


def _role_check(facts: NodeFacts) -> Check:
    role = (facts.worker_role or "").strip().lower()
    if role in {"serving", "all"}:
        return Check("worker role", Severity.PASS, f"DOCIE_WORKER_ROLE={role!r}")
    return Check(
        "worker role",
        Severity.FAIL,
        f"DOCIE_WORKER_ROLE={role!r} does not include the serving role",
        "This process registers no serving functions, so it will never deploy, "
        "reconcile or publish capacity. Set DOCIE_WORKER_ROLE=serving.",
    )


def _home_check(facts: NodeFacts) -> Check:
    if facts.serving_home_writable:
        return Check("serving home", Severity.PASS, facts.serving_home or "(default)")
    return Check(
        "serving home",
        Severity.FAIL,
        f"{facts.serving_home!r} is not writable",
        "deployments.json, the node id and the blob store all live here. "
        "Without it this node cannot remember what it is running, or even what "
        "it is called.",
    )


def _node_id_check(facts: NodeFacts) -> Check:
    if not facts.node_id:
        return Check(
            "node id",
            Severity.FAIL,
            "no node id could be resolved",
            "Placements are keyed by node. Without an id this node's capacity "
            "row and its placements cannot be told apart from another's.",
        )
    others = facts.other_node_ids
    if others is not None and facts.node_id in others:
        return Check(
            "node id",
            Severity.FAIL,
            f"{facts.node_id!r} is ALSO published by another node",
            "ModelPlacement.name is a primary key, so two nodes sharing an id "
            "overwrite each other's rows: one node's replica becomes "
            "unroutable while still holding its RAM. Give this node a unique "
            "DOCIE_SERVING_NODE_ID.",
        )
    source = {
        "env": "from DOCIE_SERVING_NODE_ID",
        "persisted": "persisted in the serving home",
        "generated": "generated and persisted just now",
    }.get(facts.node_id_source, "")
    return Check(
        "node id",
        Severity.PASS,
        f"{facts.node_id} ({source})" if source else facts.node_id,
    )


def _advertise_check(facts: NodeFacts) -> Check:
    host = (facts.advertise_host or "").strip()
    if host.lower() in UNROUTABLE_ADVERTISE_HOSTS:
        return Check(
            "advertise host",
            Severity.FAIL,
            f"DOCIE_SERVING_ADVERTISE_HOST={host!r}",
            "Every deploy here records its endpoint as http://{this}:{port}/v1, "
            "and the central api resolves that string in ITS network -- where "
            "this value is the api's own container. Requests for this node's "
            "models would be answered by the wrong machine. Use this node's "
            "private-network address.",
        )
    resolved = facts.advertise_resolved
    if resolved is None:
        return Check(
            "advertise host",
            Severity.WARN,
            f"{host!r} does not resolve from this node",
            "It may still resolve in the central api's network, which is the "
            "network that matters -- but a name that resolves nowhere here is "
            "usually a name that resolves to the wrong thing there.",
        )
    if facts.local_addresses and not (set(resolved) & facts.local_addresses):
        return Check(
            "advertise host",
            Severity.WARN,
            f"{host!r} resolves to {', '.join(sorted(resolved))}, "
            f"none of which is an address of this host",
            "This is the #538 failure: a name that resolves to some OTHER "
            "machine. Deploys record it anyway, and the central api reaches "
            "that other machine for this node's models.",
        )
    return Check("advertise host", Severity.PASS, f"{host} -> {', '.join(sorted(resolved))}")


def _catalog_check(facts: NodeFacts) -> Check:
    if facts.db_reachable:
        return Check("model catalog", Severity.PASS, facts.db_detail or "reachable")
    return Check(
        "model catalog",
        Severity.FAIL,
        facts.db_detail or "the catalog could not be read",
        "The catalog is how this node publishes capacity, records placements "
        "and reads desired state. Without it the node still serves what is "
        "already loaded, but it is invisible to the fleet and converges on "
        "nothing.",
    )


def _clock_check(facts: NodeFacts) -> Check:
    skew = facts.db_clock_skew_s
    if skew is None:
        return Check("clocks", Severity.INFO, "clock comparison unavailable")
    return Check(
        "clocks",
        Severity.INFO,
        f"the database clock is {skew:+.1f}s from this node's",
        # Deliberately not a WARN against the staleness window. Staleness
        # compares a timestamp this node writes against the clock of the
        # process READING it -- the central api -- so the pair that matters is
        # node-to-api, and the database is only a convenient third reference.
        "Liveness and the drain grace period compare timestamps written here "
        "against another machine's clock. If the central api's clock differs "
        "similarly, this node will look stale while healthy (or fresh while "
        "gone). Worth checking NTP on both if this number is large.",
    )


def _inngest_check(facts: NodeFacts) -> Check:
    url = facts.inngest_base_url or "(unset)"
    if facts.inngest_reachable is None:
        return Check("inngest", Severity.INFO, f"{url}: not probed")
    if facts.inngest_reachable:
        return Check("inngest", Severity.PASS, url)
    return Check(
        "inngest",
        Severity.WARN,
        f"{url} is not reachable from this node",
        "Lifecycle events (seed, delete, load) are delivered over Connect, "
        "which is outbound from here. Until this is reachable the node "
        "receives no events -- though desired state, which is read from the "
        "catalog, still converges.",
    )


def _desired_apply_check(facts: NodeFacts) -> Check:
    if facts.desired_apply:
        return Check("desired-state apply", Severity.PASS, "DOCIE_SERVING_DESIRED_APPLY is on")
    return Check(
        "desired-state apply",
        Severity.WARN,
        "DOCIE_SERVING_DESIRED_APPLY is off, so this node computes plans and "
        "carries none of them out",
        "Three things silently do nothing on this node: a deploy targeted at "
        "it, an auto-placed deploy that picks it, and the unload half of a "
        "drain -- so a drained node leaves the rotation but keeps its memory. "
        "Routing exclusion and capacity publishing work regardless. Set "
        "DOCIE_SERVING_DESIRED_APPLY=1 once the node is otherwise healthy.",
    )


def _ports_check(facts: NodeFacts) -> Check:
    start, end = facts.port_range
    if start > end:
        return Check(
            "port window",
            Severity.FAIL,
            f"DOCIE_SERVING_PORT_RANGE_START={start} is above END={end}",
            "No port can be allocated, so every deploy on this node fails.",
        )
    held = facts.ports_in_use
    if held is None:
        return Check(
            "port window",
            Severity.INFO,
            f"{start}-{end} (not probed; pass --probe-ports inside the "
            f"serving container to check it)",
        )
    if held:
        return Check(
            "port window",
            Severity.WARN,
            f"{start}-{end}, with {len(held)} already in use: "
            f"{', '.join(str(port) for port in held)}",
            "The allocator skips a bound port, so this only costs capacity -- "
            "unless the port is held by something OUTSIDE this container, in "
            "which case the published range and the allocatable range "
            "disagree and a deploy can record an endpoint nothing reaches. "
            "Only trust this when the doctor ran INSIDE the serving container: "
            "on the host it probes a different network namespace and reports "
            "conflicts the runtime would never see.",
        )
    return Check("port window", Severity.PASS, f"{start}-{end}, all free")


def _blobs_check(facts: NodeFacts) -> Check:
    if facts.catalog_models is None or facts.store_models is None:
        return Check(
            "blob store",
            Severity.INFO,
            "cannot compare this node's blobs against the catalog",
        )
    missing = sorted(set(facts.catalog_models) - set(facts.store_models))
    if not missing:
        return Check(
            "blob store",
            Severity.PASS,
            f"holds all {len(facts.catalog_models)} catalogued model(s)",
        )
    return Check(
        "blob store",
        Severity.INFO,
        f"{len(missing)} catalogued model(s) are not on this node: "
        f"{', '.join(missing)}",
        "The catalog is global and the blob store is node-local, so this is "
        "normal rather than wrong -- but a deploy of one of these aimed at "
        "this node is refused until it is seeded here.",
    )


#: Every check, in the order an operator should read them: what the node is,
#: then whether the fleet can see it, then what it will actually do.
_CHECKS = (
    _role_check,
    _home_check,
    _node_id_check,
    _advertise_check,
    _catalog_check,
    _inngest_check,
    _desired_apply_check,
    _ports_check,
    _blobs_check,
    _clock_check,
)


def evaluate(facts: NodeFacts) -> list[Check]:
    """Judge ``facts``. Pure: no node, no database, no socket."""
    return [check(facts) for check in _CHECKS]


def worst(checks: Sequence[Check]) -> Severity:
    """The most serious severity present, PASS when there is nothing worse."""
    order = (Severity.FAIL, Severity.WARN, Severity.INFO, Severity.PASS)
    present = {check.severity for check in checks}
    for severity in order:
        if severity in present:
            return severity
    return Severity.PASS


def _bool_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _local_addresses() -> frozenset[str]:
    """This host's own addresses, best-effort."""
    found: set[str] = {"127.0.0.1", "::1"}
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            found.add(str(info[4][0]))
    except OSError:
        pass
    return frozenset(found)


def _resolve(host: str) -> tuple[str, ...] | None:
    try:
        return tuple(sorted({str(info[4][0]) for info in socket.getaddrinfo(host, None)}))
    except OSError:
        return None


def _ports_held(bind_host: str, start: int, end: int, *, limit: int = 32) -> tuple[int, ...]:
    """Which ports in the window are already bound, probed by binding them.

    Capped: a wide window would mean hundreds of syscalls for a check whose
    purpose is to notice that something else is squatting the range.
    """
    held: list[int] = []
    host = bind_host if bind_host not in {"", "0.0.0.0"} else "0.0.0.0"  # noqa: S104
    for port in range(start, min(end, start + limit - 1) + 1):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
        except OSError:
            held.append(port)
        finally:
            probe.close()
    return tuple(held)


def collect_facts(*, probe_ports: bool = False, probe_inngest: bool = True) -> NodeFacts:
    """Gather what :func:`evaluate` judges, from this machine.

    Every probe is contained: an unreachable database or a DNS failure leaves
    its own fields unknown and lets every other check report. A doctor that
    says nothing because one probe failed is worse than no doctor.

    ``probe_ports`` is off by default because binding the deploy window only
    means something in the serving container's own network namespace. Run from
    the host it would report ports held on unrelated interfaces as conflicts
    the runtime will never see, and a check that cries wolf on first use costs
    the credibility of every check beside it.
    """
    from docie_bench.serving.paths import serving_home
    from docie_bench.settings import get_settings

    settings = get_settings()
    home = serving_home()
    writable = False
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".preflight"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        writable = True
    except OSError:
        writable = False

    node_id: str | None = None
    node_id_source = ""
    try:
        from docie_bench.serving.node_identity import (
            NODE_ID_ENV_VAR,
            node_id_path,
            resolve_node_id,
        )

        # Established BEFORE resolving, because resolving a fresh node mints and
        # persists an id -- so afterwards "generated" and "persisted" look the
        # same on disk.
        if os.getenv(NODE_ID_ENV_VAR, "").strip():
            node_id_source = "env"
        elif node_id_path().exists():
            node_id_source = "persisted"
        else:
            node_id_source = "generated"
        node_id = resolve_node_id()
    except Exception:  # noqa: BLE001 - an unresolvable id is itself a finding
        node_id = None
        node_id_source = ""

    advertise = str(getattr(settings, "serving_advertise_host", "") or "")
    bind_host = os.getenv("DOCIE_SERVING_BIND_HOST", "0.0.0.0")  # noqa: S104
    start = int(getattr(settings, "serving_port_range_start", 8088))
    end = int(getattr(settings, "serving_port_range_end", 8188))

    db_reachable = False
    db_detail: str | None = None
    skew: float | None = None
    catalog_models: tuple[str, ...] | None = None
    store_models: tuple[str, ...] | None = None
    other_nodes: tuple[str, ...] | None = None
    try:
        import docie_bench.storage.db as db
        from docie_bench.serving.catalog import ModelCatalog

        with db.session_scope() as session:
            if session is None:
                raise RuntimeError("DATABASE_URL is not configured")
            from sqlalchemy import text as sa_text

            now = session.execute(sa_text("SELECT CURRENT_TIMESTAMP")).scalar()
        db_reachable = True
        remote = _as_datetime(now)
        if remote is not None:
            skew = (remote - dt.datetime.now(dt.UTC)).total_seconds()
        catalog = ModelCatalog()
        catalog_models = tuple(str(row["name"]) for row in catalog.list())
        rows = catalog.list_node_snapshots()
        other_nodes = tuple(
            str(row.get("node_id") or "")
            for row in rows
            if str(row.get("node_id") or "") and str(row.get("node_id") or "") != node_id
        )
        db_detail = f"reachable; {len(rows)} node(s) known, {len(catalog_models)} model(s)"
    except Exception as exc:  # noqa: BLE001 - every failure here is a finding
        db_detail = f"{type(exc).__name__}: {exc}"

    try:
        from docie_bench.serving.model_store import ModelStore

        # The root is required, and every other caller passes this same path
        # (functions.py, serving_api.py, cli.py, resources.py). Omitting it
        # raised a TypeError that the handler below turned into "unknown",
        # so the comparison silently never ran on any node.
        store_models = tuple(
            sorted(entry.name for entry in ModelStore(home / "models").list())
        )
    except Exception:  # noqa: BLE001 - an unreadable store leaves the compare unknown
        # Logged rather than swallowed: the first version of this hid a
        # programming error as a legitimate degraded result, and a check that
        # reports "we cannot tell" when it is actually broken is worse than one
        # that fails, because nothing ever looks wrong.
        logger.debug("could not read this node's blob store", exc_info=True)
        store_models = None

    inngest_url = os.getenv("INNGEST_BASE_URL") or None
    inngest_reachable: bool | None = None
    if probe_inngest and inngest_url:
        inngest_reachable = _reachable(inngest_url)

    return NodeFacts(
        worker_role=os.getenv("DOCIE_WORKER_ROLE", "all"),
        serving_home=str(home),
        serving_home_writable=writable,
        node_id=node_id,
        node_id_source=node_id_source,
        advertise_host=advertise,
        advertise_resolved=_resolve(advertise) if advertise else None,
        local_addresses=_local_addresses(),
        bind_host=bind_host,
        port_range=(start, end),
        ports_in_use=_ports_held(bind_host, start, end) if probe_ports else None,
        desired_apply=_bool_env("DOCIE_SERVING_DESIRED_APPLY"),
        db_reachable=db_reachable,
        db_detail=db_detail,
        db_clock_skew_s=skew,
        inngest_base_url=inngest_url,
        inngest_reachable=inngest_reachable,
        catalog_models=catalog_models,
        store_models=store_models,
        other_node_ids=other_nodes,
    )


def _as_datetime(value: object) -> dt.datetime | None:
    """A database CURRENT_TIMESTAMP as an aware datetime, or None.

    Driver-dependent: psycopg returns a datetime, SQLite returns a string. The
    first version of this only handled the datetime, so the clock comparison
    silently never ran under SQLite -- working in production and nowhere a test
    could see it.
    """
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.UTC)
    if isinstance(value, str) and value.strip():
        try:
            parsed = dt.datetime.fromisoformat(value.strip().replace(" ", "T"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)
    return None


def _reachable(url: str, *, timeout: float = 3.0) -> bool:
    """Whether a TCP connection to ``url``'s host:port succeeds."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


__all__ = [
    "UNROUTABLE_ADVERTISE_HOSTS",
    "Check",
    "NodeFacts",
    "Severity",
    "collect_facts",
    "evaluate",
    "worst",
]
