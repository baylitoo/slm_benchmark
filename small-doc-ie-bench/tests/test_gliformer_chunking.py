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
    split_into_pages,
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


def test_the_tokenizer_is_preferred_over_the_backbone_position_table() -> None:
    # _Config declares only max_position_embeddings (1024) and _Tokenizer
    # declares model_max_length (4096). The backbone's table is the last
    # resort, so the tokenizer wins.
    model = type("M", (), {"config": _Config(), "tokenizer": _Tokenizer()})()
    assert model_input_window(model) == 4096


# The two published checkpoints, verbatim from their gliner_config.json. Both
# report max_position_embeddings 512, which belongs to the DeBERTa backbone's
# absolute position table and is NOT what the encoder accepts.
@pytest.mark.parametrize(
    ("checkpoint", "max_len", "expected"),
    [("gliformer-base-v1", 16384, 16384), ("gliformer-large-v1", 8192, 8192)],
)
def test_the_published_checkpoints_report_their_real_window(
    checkpoint: str, max_len: int, expected: int
) -> None:
    config = type("C", (), {"max_len": max_len, "max_position_embeddings": 512})()
    model = type("M", (), {"config": config})()
    assert model_input_window(model) == expected


def test_the_backbone_position_table_never_wins_over_max_len() -> None:
    # Reading it first chunked a document into sixteen to thirty-two times more
    # pieces than needed, each seam another call and another place to cut a
    # record in half.
    config = type("C", (), {"max_len": 8192, "max_position_embeddings": 512})()
    assert model_input_window(type("M", (), {"config": config})()) == 8192


def test_the_backbone_table_is_still_used_when_nothing_else_is_declared() -> None:
    config = type("C", (), {"max_position_embeddings": 512})()
    assert model_input_window(type("M", (), {"config": config})()) == 512


def test_max_len_on_the_model_itself_is_accepted() -> None:
    assert model_input_window(type("M", (), {"max_len": 2048})()) == 2048


def test_a_boolean_is_not_a_window() -> None:
    config = type("C", (), {"max_len": True, "max_position_embeddings": 512})()
    assert model_input_window(type("M", (), {"config": config})()) == 512


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


# ── pages are the unit a document declares ───────────────────────────────────


def _page(number: int, lines: int) -> str:
    body = "\n".join(f"page {number} ligne {i} avec du texte" for i in range(lines))
    return f"[page {number}]\n{body}"


def test_a_document_is_divided_on_its_own_page_boundaries() -> None:
    document = "\n".join([_page(1, 10), _page(2, 10), _page(3, 10)])
    chunks = split_for_window(document, words, 60)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.lstrip().startswith("[page ")


def test_a_sparse_page_rides_along_with_the_next_one() -> None:
    # A cover sheet or a page holding one table should not cost its own
    # inference call; grouping whole pages up to the budget handles it.
    document = "\n".join([_page(1, 2), _page(2, 2), _page(3, 2)])
    chunks = split_for_window(document, words, 60)
    assert len(chunks) == 1
    assert "[page 1]" in chunks[0]
    assert "[page 3]" in chunks[0]


def test_pages_are_never_cut_when_they_fit() -> None:
    document = "\n".join([_page(1, 8), _page(2, 8)])
    chunks = split_for_window(document, words, 70)
    assert len(chunks) == 2
    assert chunks[0].startswith("[page 1]")
    assert chunks[1].startswith("[page 2]")


def test_only_a_page_too_large_to_keep_whole_is_cut() -> None:
    document = "\n".join([_page(1, 3), _page(2, 40), _page(3, 3)])
    chunks = split_for_window(document, words, 120)
    from_page_two = [c for c in chunks if "[page 2]" in c]
    assert len(from_page_two) > 1
    # The pages either side stay whole.
    assert any(c.startswith("[page 1]") and "[page 2]" not in c for c in chunks)


def test_every_fragment_of_a_cut_page_still_says_which_page_it_is() -> None:
    document = "\n".join([_page(1, 60), _page(2, 2)])
    for chunk in split_for_window(document, words, 80):
        assert chunk.splitlines()[0].startswith("[page ")


def test_the_ceiling_holds_once_the_marker_is_added_back() -> None:
    # The marker is prepended after the body is split, so its cost has to come
    # out of the budget first or a fragment that just fitted no longer does.
    document = "\n".join([_page(1, 60), _page(2, 60)])
    for budget in (30, 50, 80, 200):
        assert all(words(chunk) <= budget for chunk in split_for_window(document, words, budget))


def test_the_ceiling_holds_over_random_paginated_documents() -> None:
    rng = random.Random(5)  # noqa: S311 - test fixture, not crypto
    for _ in range(200):
        pages = [_page(i, rng.randint(1, 30)) for i in range(1, rng.randint(2, 7))]
        budget = rng.randint(20, 200)
        for chunk in split_for_window("\n".join(pages), words, budget):
            assert words(chunk) <= budget


def test_a_paginated_document_that_fits_is_still_sent_whole() -> None:
    document = "\n".join([_page(1, 5), _page(2, 5)])
    assert split_for_window(document, words, 10_000) == [document]


def test_text_with_no_page_markers_falls_back_to_lines() -> None:
    document = "\n".join("une ligne de texte sans marqueur" for _ in range(40))
    chunks = split_for_window(document, words, 60)
    assert len(chunks) > 1
    assert all(words(chunk) <= 60 for chunk in chunks)


def test_split_into_pages_keeps_the_marker_with_its_page() -> None:
    pages = split_into_pages("\n".join([_page(1, 2), _page(2, 2)]))
    assert len(pages) == 2
    assert pages[0].startswith("[page 1]")
    assert pages[1].startswith("[page 2]")


def test_split_into_pages_returns_one_piece_for_unmarked_text() -> None:
    assert split_into_pages("une ligne\nune autre") == ["une ligne\nune autre"]
