"""GLiClass: zero-shot multi-label classification through the encoder shim.

The distinguishing property, and what these tests protect: GLiClass SCORES
labels, it does not locate them. It has no spans and no structure head, so the
shim has to answer it on its own branch rather than calling a ``predict`` it
deliberately does not implement.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from docie_bench.encoders.server import (
    GliClassBackend,
    build_backend,
    create_encoder_app,
)
from docie_bench.serving.arch_registry import RepoSignals, _is_gliclass
from docie_bench.serving.model_store import FAMILIES

# GLiClass sits on a plain DeBERTa, so this is what the Hub reports as its
# architecture -- which says nothing about GLiClass. That is the point.
DEBERTA = "debertav2forsequenceclassification"


class FakeClassifier:
    """Stands in for ``ZeroShotClassificationPipeline``.

    Returns the pipeline's real shape — a list (one entry per input text) of
    lists of ``{"label", "score"}`` — so the backend's unwrapping is exercised
    rather than assumed.
    """

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str], float]] = []

    def __call__(self, text: str, labels: list[str], threshold: float = 0.5) -> Any:
        self.calls.append((text, list(labels), threshold))
        return [
            [
                {"label": label, "score": self.scores[label]}
                for label in labels
                if self.scores.get(label, 0.0) >= threshold
            ]
        ]


def _backend(scores: dict[str, float], *, multi_label: bool = True) -> GliClassBackend:
    backend = GliClassBackend.__new__(GliClassBackend)  # bypass model loading
    backend.model_id = "knowledgator/gliclass-v4-edge"
    backend.multi_label = multi_label
    backend._pipeline = FakeClassifier(scores)
    return backend


def test_scores_come_back_sorted_highest_first() -> None:
    backend = _backend({"invoice": 0.91, "contract": 0.62, "receipt": 0.55})

    scored = backend.classify_labels("...", ["contract", "receipt", "invoice"], 0.5)

    assert [row["label"] for row in scored] == ["invoice", "contract", "receipt"]
    assert scored[0]["score"] == pytest.approx(0.91)


def test_multi_label_keeps_every_label_over_threshold() -> None:
    # The point of multi-label: a document can be several things at once.
    backend = _backend({"invoice": 0.91, "urgent": 0.78, "draft": 0.10})

    scored = backend.classify_labels("...", ["invoice", "urgent", "draft"], 0.5)

    assert [row["label"] for row in scored] == ["invoice", "urgent"]


def test_single_label_collapses_to_the_winner() -> None:
    # Mutually exclusive categories: two labels over the threshold is still one
    # answer, not two.
    backend = _backend({"invoice": 0.91, "urgent": 0.78}, multi_label=False)

    scored = backend.classify_labels("...", ["invoice", "urgent"], 0.5)

    assert [row["label"] for row in scored] == ["invoice"]


def test_a_request_may_override_the_deployment_default() -> None:
    backend = _backend({"a": 0.9, "b": 0.8}, multi_label=True)

    assert len(backend.classify_labels("...", ["a", "b"], 0.5, multi_label=False)) == 1
    assert len(backend.classify_labels("...", ["a", "b"], 0.5, multi_label=True)) == 2


def test_the_threshold_reaches_the_pipeline() -> None:
    backend = _backend({"a": 0.9})

    backend.classify_labels("doc text", ["a"], 0.75)

    assert backend._pipeline.calls[-1] == ("doc text", ["a"], 0.75)


def test_it_does_not_implement_predict() -> None:
    # Not an oversight. There are no spans to return, and a shim that found a
    # `predict` here would hand downstream code offsets that mean nothing.
    assert not hasattr(_backend({}), "predict")
    assert GliClassBackend.classification_only is True


# --- dispatch -----------------------------------------------------------


def test_auto_detects_gliclass_from_the_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    built: dict[str, Any] = {}

    class Stub:
        def __init__(self, model_id: str) -> None:
            built["model_id"] = model_id

    monkeypatch.setattr("docie_bench.encoders.server.GliClassBackend", Stub)

    build_backend("knowledgator/gliclass-v4-edge", "auto")

    assert built["model_id"] == "knowledgator/gliclass-v4-edge"


def test_an_unknown_backend_names_gliclass_among_the_options() -> None:
    with pytest.raises(ValueError, match="gliclass"):
        build_backend("some/model", "not-a-backend")


# --- family and detection ----------------------------------------------


def test_the_family_is_an_analyzer_with_no_extraction_head() -> None:
    family = FAMILIES["encoder_gliclass"]

    assert family.analyzer is True
    assert family.encoder_backend == "gliclass"
    # It classifies; it does not answer a schema. Claiming otherwise would
    # route extraction requests to a model that cannot serve them.
    assert family.structured_extraction is False


@pytest.mark.parametrize(
    "signals",
    [
        RepoSignals(arch=DEBERTA, repo_id="knowledgator/gliclass-v4-edge"),
        RepoSignals(arch=DEBERTA, repo_id="x/y", library_name="gliclass"),
        RepoSignals(arch=DEBERTA, repo_id="x/y", tags=("gliclass",)),
    ],
)
def test_detected_by_how_the_repo_describes_itself(signals: RepoSignals) -> None:
    # The backbone is a plain DeBERTa, so the architecture string says nothing
    # about GLiClass -- same problem GLiFormer has, solved the same way.
    assert _is_gliclass(signals) is True


def test_not_confused_with_the_extractor_families() -> None:
    gliformer = RepoSignals(arch="", repo_id="knowledgator/gliformer-base-v1")
    assert _is_gliclass(gliformer) is False
    gliner2 = RepoSignals(arch="gliner2model", repo_id="fastino/GLiNER2-Guardrails")
    assert _is_gliclass(gliner2) is False


# --- through the shim ---------------------------------------------------


def test_the_shim_answers_with_label_scores_not_entities() -> None:
    app = create_encoder_app(
        model_id="knowledgator/gliclass-v4-edge",
        backend=_backend({"invoice": 0.9, "receipt": 0.2}),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "knowledgator/gliclass-v4-edge",
                "messages": [{"role": "user", "content": "Facture n 42"}],
                "labels": ["invoice", "receipt"],
                "threshold": 0.5,
            },
        )

    assert response.status_code == 200
    import json

    payload = json.loads(response.json()["choices"][0]["message"]["content"])
    assert [row["label"] for row in payload["labels"]] == ["invoice"]
    # No entity list: nothing was located.
    assert "entities" not in payload
