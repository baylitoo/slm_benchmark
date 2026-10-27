"""A store name becomes a path, so it is validated before it becomes one.

`POST /v1/studio/seed-hf` takes a store name. It was declared
`name: str | None = None` with no constraint, and `seed_hf_job` joined it under
the store root to stage the download:

    tmp_dir = store.root / ".hf-downloads" / name

The store root is `serving_home()/models`, and `serving_home()` itself holds
`node_id`, `deployments.json` and the applied-state file.

The staging directory sits one level below the store root, so a single `..`
only lands back inside it; **two** reach `serving_home()` and the files this
node's identity and convergence depend on. Either way the name was never
usable as a directory, and nothing checked it before the join.

Authenticated only, and a file write rather than code execution. But the write
lands on exactly the files a node cannot afford to lose, and the name reaches
the path join before any of the store's own containment checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from docie_bench.inngest.studio_api.deploy import SeedHfRequest
from docie_bench.serving.hf_hub import default_store_name
from docie_bench.serving.model_store import ModelStoreError, validate_store_name

# Every shape that turns a name into a path, plus the ones that are simply not
# usable as a directory or a catalog key.
REFUSED = [
    "../node_id",
    "../../node_id",
    "../../deployments.json",
    "../../etc/cron.d/x",
    "..",
    ".",
    "a/b",
    "a\\b",
    "/absolute",
    ".hidden",
    "-leading-dash",
    "",
    "trailing/",
    "x\n../y",
]

# Names the system itself produces or an operator reasonably types.
ACCEPTED = [
    "lfm2.5-350m",
    "nuextract3",
    "embeddinggemma-2",
    "gliner-multi-pii",
    "qwen3-vl-2b",
    "lfm2.5-350m-instruct",
    "model_with_underscores",
    "a",
]


@pytest.mark.parametrize("name", ACCEPTED)
def test_the_names_the_system_actually_uses_are_accepted(name: str) -> None:
    """A validator that rejected the project's own generated names would break
    seeding entirely, which is the failure mode worth checking first."""
    assert validate_store_name(name) == name


@pytest.mark.parametrize(
    "repo",
    [
        "LiquidAI/LFM2.5-350M-Instruct-GGUF",
        "numind/NuExtract-2.0-2B-GGUF",
        "unsloth/Qwen3-VL-2B-Instruct-GGUF",
    ],
)
def test_every_derived_default_name_passes(repo: str) -> None:
    """`default_store_name` is what fills the field when a caller omits it, so
    the validator has to admit everything it emits."""
    assert validate_store_name(default_store_name(repo))


@pytest.mark.parametrize("name", REFUSED)
def test_a_name_that_could_become_a_path_is_refused(name: str) -> None:
    with pytest.raises(ModelStoreError):
        validate_store_name(name)


def test_a_valid_name_cannot_escape_the_join_it_is_used_in() -> None:
    """The property the validator exists for, stated directly rather than
    through the job: anything it accepts stays inside the root when joined,
    which is exactly how `seed_hf_job` stages a download."""
    root = Path("/srv/serving/models").resolve()

    for name in ACCEPTED:
        staged = (root / ".hf-downloads" / name).resolve()
        assert root in staged.parents, name


def test_a_refused_name_would_have_escaped() -> None:
    """The negative half: these are refused BECAUSE the join does not contain
    them. Without this the test above would pass against a validator that
    accepted nothing useful."""
    root = Path("/srv/serving/models").resolve()
    escaped = [
        name
        for name in REFUSED
        if name and root not in (root / ".hf-downloads" / name).resolve().parents
    ]

    # Two parent segments are what it takes: the staging dir is one level
    # below the store root, so a single `..` lands back INSIDE the root
    # (`models/node_id`). `../../node_id` is the one that reaches the node
    # identity file beside the store.
    assert "../../node_id" in escaped
    assert "../../deployments.json" in escaped
    assert "../../etc/cron.d/x" in escaped
    assert "../node_id" not in escaped  # inside the root, but still not a name


# --- the API boundary -----------------------------------------------------


def test_the_request_model_rejects_a_traversal_name() -> None:
    """Rejected on the request, so it never becomes a queued job that fails
    after downloading several gigabytes."""
    with pytest.raises(ValidationError):
        SeedHfRequest(repo="owner/repo", name="../node_id")


def test_the_request_model_still_accepts_an_omitted_name() -> None:
    """Omitted is the common case -- the name is then derived from the repo."""
    assert SeedHfRequest(repo="owner/repo").name is None


def test_the_request_model_accepts_a_real_name() -> None:
    assert SeedHfRequest(repo="owner/repo", name="lfm2.5-350m").name == "lfm2.5-350m"


def test_the_request_pattern_anchors_both_ends() -> None:
    """pydantic compiles the pattern with the Rust regex crate, where a bare
    `$` would still match before a trailing newline in multiline mode. It is
    not multiline here, and this pins that."""
    with pytest.raises(ValidationError):
        SeedHfRequest(repo="owner/repo", name="ok\n../escape")


def test_the_job_validates_too_because_events_have_other_callers() -> None:
    """The request model is not the only way into `seed_hf_job` -- collection
    and batch paths emit the same event -- so the job validates the name again
    rather than trusting its caller."""
    import inspect

    from docie_bench.inngest import functions

    source = inspect.getsource(functions)
    guard = source.index("validate_store_name(name)")
    staging = source.index('store.root / ".hf-downloads" / name')

    assert guard < staging, "the name must be validated before it becomes a path"
