"""A document past the encoder's input window is structured in pieces."""

import random
from typing import Any

import pytest

from docie_bench.encoders.server import (
    DEFAULT_STRUCTURE_WINDOW_TOKENS,
    GliformerBackend,
    merge_records,
    model_input_window,
    split_for_window,
)


def words(text: str) -> int:
    return len(text.split())


# ── the window is read off the model, not guessed ────────────────────────────


class _Config:
    max_position_embeddings = 1024


class _Tokenizer:
    model_max_length = 4096

    def encode(self, text: str) -> list[int]:
        return text.split()


def test_the_model_config_declares_the_window() -> None:
    model = type("M", (), {"config": _Config(), "tokenizer": _Tokenizer()})()
    assert model_input_window(model) == 1024


def test_the_tokenizer_answers_when_the_config_does_not() -> None:
    model = type("M", (), {"tokenizer": _Tokenizer()})()
    assert model_input_window(model) == 4096


def test_a_sentinel_length_reads_as_unknown_rather_than_unbounded() -> None:
    # transformers writes a billion-scale sentinel when a tokenizer declares no
    # limit; taking that literally would never chunk anything.
    tokenizer = type("T", (), {"model_max_length": 1_000_000_000})()
    model = type("M", (), {"tokenizer": tokenizer})()
    assert model_input_window(model) is None


def test_a_model_that_declares_nothing_falls_back() -> None:
    assert model_input_window(object()) is None
    assert DEFAULT_STRUCTURE_WINDOW_TOKENS > 0


# ── splitting ────────────────────────────────────────────────────────────────


def test_a_document_that_fits_is_not_split() -> None:
    text = "une ligne\nune autre"
    assert split_for_window(text, words, 100) == [text]


def test_no_piece_exceeds_the_budget() -> None:
    text = "\n".join(f"ligne {i} avec quelques mots" for i in range(12))
    chunks = split_for_window(text, words, 12)
    assert len(chunks) > 1
    assert all(words(chunk) <= 12 for chunk in chunks)


def test_the_seam_repeats_a_few_lines_so_a_record_is_seen_whole() -> None:
    text = "\n".join(f"l{i}" for i in range(10))
    chunks = split_for_window(text, words, 4, overlap_lines=2)
    assert len(chunks) > 1
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert set(earlier.splitlines()) & set(later.splitlines())


def test_the_overlap_is_dropped_before_the_budget_is() -> None:
    # Carrying two lines would push the next piece over; the ceiling wins.
    text = "\n".join("mot " * 4 for _ in range(6))
    chunks = split_for_window(text, words, 8, overlap_lines=2)
    assert all(words(chunk) <= 8 for chunk in chunks)


def test_a_single_line_over_budget_is_split_rather_than_dropped() -> None:
    text = "mot " * 50
    chunks = split_for_window(text.strip(), words, 10)
    assert all(words(chunk) <= 10 for chunk in chunks)
    assert sum(words(chunk) for chunk in chunks) == 50


def test_the_ceiling_holds_over_random_documents() -> None:
    rng = random.Random(3)  # noqa: S311 - test fixture, not crypto
    for _ in range(200):
        lines = [" ".join("w" for _ in range(rng.randint(1, 9))) for _ in range(rng.randint(1, 40))]
        budget = rng.randint(3, 25)
        chunks = split_for_window("\n".join(lines), words, budget)
        assert all(words(chunk) <= budget for chunk in chunks)
        assert sum(len(chunk.splitlines()) for chunk in chunks) >= len(lines)


# ── merging ──────────────────────────────────────────────────────────────────


def test_a_list_field_gathers_what_each_chunk_saw() -> None:
    merged = merge_records(
        [
            {"experience": [{"title": "Dev"}]},
            {"experience": [{"title": "Lead"}]},
        ]
    )
    assert merged["experience"] == [{"title": "Dev"}, {"title": "Lead"}]


def test_the_same_record_seen_twice_at_a_seam_appears_once() -> None:
    merged = merge_records(
        [
            {"experience": [{"title": "Dev"}]},
            {"experience": [{"title": "Dev"}, {"title": "Lead"}]},
        ]
    )
    assert merged["experience"] == [{"title": "Dev"}, {"title": "Lead"}]


def test_a_scalar_takes_the_first_chunk_that_answered() -> None:
    merged = merge_records([{"full_name": None}, {"full_name": "Ada"}, {"full_name": "Grace"}])
    assert merged["full_name"] == "Ada"


def test_empty_answers_never_shadow_a_later_one() -> None:
    merged = merge_records([{"total": ""}, {"total": {"amount": "10", "currency": "EUR"}}])
    assert merged["total"] == {"amount": "10", "currency": "EUR"}


def test_a_non_record_answer_is_ignored() -> None:
    assert merge_records([{"a": 1}, "nonsense", None]) == {"a": 1}  # type: ignore[list-item]


# ── the backend wires the two together ───────────────────────────────────────


class _FakeModel:
    tokenizer = _Tokenizer()

    def __init__(self) -> None:
        self.seen: list[str] = []

    def structure(self, text: str, schema: Any, validate_output: bool = True) -> Any:
        self.seen.append(text)
        index = len(self.seen)
        return {"resume": [{"full_name": "Ada", "experience": [{"title": f"Job {index}"}]}]}


def _backend(window: int) -> GliformerBackend:
    backend = GliformerBackend.__new__(GliformerBackend)
    backend.model_id = "knowledgator/gliformer-base-v1"
    backend._model = _FakeModel()
    backend.input_window = window
    return backend


def test_a_long_document_is_structured_in_pieces_and_folded_into_one_record() -> None:
    backend = _backend(window=6)
    text = "\n".join(f"ligne {i} du cv" for i in range(10))
    result = backend.structure(text, {"resume": object})
    assert len(backend._model.seen) > 1
    records = result["resume"]
    assert len(records) == 1
    assert records[0]["full_name"] == "Ada"
    # Every chunk's experiences survive; without this the tail of the document
    # is silently never read.
    assert len(records[0]["experience"]) == len(backend._model.seen)


def test_a_short_document_is_sent_whole_and_answered_verbatim() -> None:
    backend = _backend(window=1000)
    result = backend.structure("court", {"resume": object})
    assert backend._model.seen == ["court"]
    assert result["resume"][0]["experience"] == [{"title": "Job 1"}]


def test_token_counting_falls_back_without_a_tokenizer() -> None:
    backend = _backend(window=10)
    backend._model = type("M", (), {})()
    assert backend.count_tokens("x" * 40) == 10


@pytest.mark.parametrize("broken", [ValueError("nope"), RuntimeError("boom")])
def test_a_failing_tokenizer_never_fails_the_request(broken: Exception) -> None:
    class _Broken:
        def encode(self, text: str) -> list[int]:
            raise broken

    backend = _backend(window=10)
    backend._model = type("M", (), {"tokenizer": _Broken()})()
    assert backend.count_tokens("x" * 40) == 10
