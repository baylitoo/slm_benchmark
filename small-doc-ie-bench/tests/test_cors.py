from __future__ import annotations

from docie_bench.api import _DEFAULT_CORS_ORIGINS, parse_cors_origins


def test_default_origins_are_localhost_not_wildcard() -> None:
    # Unset env -> explicit localhost Studio origins, never "*".
    origins = parse_cors_origins(None)
    assert origins == ["http://localhost:3000", "http://127.0.0.1:3000"]
    assert "*" not in origins


def test_empty_or_blank_falls_back_to_default() -> None:
    assert parse_cors_origins("") == _DEFAULT_CORS_ORIGINS
    assert parse_cors_origins("   ") == _DEFAULT_CORS_ORIGINS
    assert parse_cors_origins(",  ,") == _DEFAULT_CORS_ORIGINS


def test_explicit_wildcard_is_preserved() -> None:
    assert parse_cors_origins("*") == ["*"]


def test_comma_separated_override_is_parsed_and_trimmed() -> None:
    origins = parse_cors_origins("https://studio.example.com, http://localhost:4000")
    assert origins == ["https://studio.example.com", "http://localhost:4000"]
