"""A deployment name becomes a log filename, so it is contained (#527 audit).

`supervisor` built the runtime's log path as `logs_dir / f"{name}.log"` with no
containment, and `runtime.start` opens it with `mkdir(parents=True)` then
`open("ab")`. The name is operator-supplied through the deploy request, so a
name carrying parent segments made the supervisor create an arbitrary directory
and append the runtime's stdout into it.

The read endpoint DID check containment, which is the other half of the problem:
reader and writer built the path independently, so any name they disagreed about
produced a log that existed and could not be read.

Mapped rather than refused, deliberately. A node restarting with a deployment
whose name predates this must still start it and still get a log; refusing
would turn a latent bad name into a node that will not come up.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from docie_bench.inngest.studio_api.deploy import DeployRequest
from docie_bench.serving.paths import deployment_log_path

# Names that genuinely leave the logs directory. These are the bug.
TRAVERSING = [
    "../../../tmp/evil",
    "../../node_id",
    "../deployments.json",
    "/absolute",
]

# Names that stay INSIDE the logs directory but are not a flat filename:
# they silently create structure under it (`logs/a/b.log`), or are not a
# usable stem at all. Not an escape, so not the security issue -- contained
# anyway, because a log the reader cannot find is a deployment nobody can
# diagnose.
NESTED = ["a/b", "a\\b", ".."]


@pytest.mark.parametrize("name", TRAVERSING + NESTED)
def test_no_name_can_put_the_log_outside_the_logs_directory(
    tmp_path: Path, name: str
) -> None:
    logs = (tmp_path / "logs").resolve()

    resolved = deployment_log_path(logs, name).resolve()

    assert logs in resolved.parents, f"{name!r} escaped to {resolved}"


@pytest.mark.parametrize("name", TRAVERSING)
def test_the_traversing_names_really_did_escape_before(
    tmp_path: Path, name: str
) -> None:
    """The negative half, and the reason the two lists are separate.

    Without it, the containment test would pass against a helper that
    mapped everything to one constant file. Only these four actually left
    the directory: `a/b` and `..` resolve inside it, so listing them as
    escapes would overstate the bug."""
    logs = (tmp_path / "logs").resolve()

    naive = (logs / f"{name}.log").resolve()

    assert logs not in naive.parents, f"{name!r} was never an escape"


@pytest.mark.parametrize("name", NESTED)
def test_the_nested_names_were_contained_but_misplaced(
    tmp_path: Path, name: str
) -> None:
    """Stated so the distinction survives: these never escaped. The reason
    to map them is that the writer would have created `logs/a/b.log` while
    the reader looked somewhere else."""
    logs = (tmp_path / "logs").resolve()

    assert deployment_log_path(logs, name).parent == logs


@pytest.mark.parametrize(
    "name",
    ["lfm2.5-350m", "lfm2.5-350m-ab12", "nuextract3", "a_b.c-d", "A1"],
)
def test_an_ordinary_name_is_left_exactly_as_it_was(tmp_path: Path, name: str) -> None:
    """Minted names are `<store-name>-<hex>`; rewriting any of them would
    orphan every existing log on an upgraded node."""
    logs = tmp_path / "logs"

    assert deployment_log_path(logs, name) == logs / f"{name}.log"


def test_an_unusable_name_still_gets_a_log(tmp_path: Path) -> None:
    """Mapped, not refused: a legacy deployment must still start and still be
    diagnosable."""
    logs = tmp_path / "logs"

    for name in ["..", "", "///"]:
        assert deployment_log_path(logs, name).name.endswith(".log")


def test_two_different_names_do_not_collide_on_one_file(tmp_path: Path) -> None:
    """Mapping must not merge two live deployments' output into one log, which
    would make both undiagnosable."""
    logs = tmp_path / "logs"

    a = deployment_log_path(logs, "svc-one")
    b = deployment_log_path(logs, "svc-two")

    assert a != b


def test_reader_and_writer_agree_on_the_path() -> None:
    """They built it independently before. The endpoint now calls the same
    helper, so a log that exists is a log that can be read."""
    import inspect

    from docie_bench.inngest import serving_api
    from docie_bench.serving import supervisor

    assert "deployment_log_path(" in inspect.getsource(serving_api.deployment_logs)
    assert "deployment_log_path(" in inspect.getsource(supervisor)
    # And neither builds it by hand any more.
    assert 'logs_dir / f"{name}.log"' not in inspect.getsource(
        serving_api.deployment_logs
    )


# --- the request boundary -------------------------------------------------


def test_a_new_deployment_cannot_be_given_an_unsafe_name() -> None:
    with pytest.raises(ValidationError):
        DeployRequest(model="lfm2.5-350m", name="../../evil")


def test_an_omitted_name_is_still_fine() -> None:
    """The common case: the name is minted from the model."""
    assert DeployRequest(model="lfm2.5-350m").name is None


def test_an_ordinary_deployment_name_is_accepted() -> None:
    assert DeployRequest(model="lfm2.5-350m", name="lfm-primary").name == "lfm-primary"
