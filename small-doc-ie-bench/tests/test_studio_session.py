from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from docie_bench import studio_session
from docie_bench.security import TenantQuotaManager
from docie_bench.settings import get_settings


@pytest.fixture
def signing_secret(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STUDIO_SESSION_SECRET", "test-session-secret")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _manager(**kwargs: object) -> TenantQuotaManager:
    defaults: dict[str, object] = {
        "api_keys": {},
        "auth_required": True,
        "requests_per_window": 100,
        "window_seconds": 60,
        "max_concurrent": 10,
    }
    defaults.update(kwargs)
    return TenantQuotaManager(**defaults)  # type: ignore[arg-type]


def test_mint_then_verify_round_trips(signing_secret: None) -> None:
    token = studio_session.mint("amine")

    assert token.startswith(studio_session.TOKEN_PREFIX)
    assert studio_session.verify(token) == "amine"


def test_a_tampered_payload_is_rejected(signing_secret: None) -> None:
    token = studio_session.mint("amine")
    body = token[len(studio_session.TOKEN_PREFIX) :]
    payload, _, signature = body.partition(".")
    # Re-sign nothing: swap the payload for another login and keep the old
    # signature. Without the HMAC check this would be a free privilege grant.
    forged = studio_session.mint("someone-else")
    other_payload = forged[len(studio_session.TOKEN_PREFIX) :].partition(".")[0]

    assert payload != other_payload
    forged_token = f"{studio_session.TOKEN_PREFIX}{other_payload}.{signature}"
    assert studio_session.verify(forged_token) is None


def test_an_expired_token_is_rejected(signing_secret: None) -> None:
    issued_at = time.time()
    token = studio_session.mint("amine", ttl_seconds=60, now=issued_at)

    assert studio_session.verify(token, now=issued_at + 59) == "amine"
    assert studio_session.verify(token, now=issued_at + 61) is None


def test_a_token_from_another_secret_is_rejected(
    signing_secret: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = studio_session.mint("amine")
    monkeypatch.setenv("STUDIO_SESSION_SECRET", "a-rotated-secret")
    get_settings.cache_clear()

    # Rotating the secret invalidates live Studio sessions, which is the
    # documented behaviour -- users sign in again, nothing else breaks.
    assert studio_session.verify(token) is None


def test_verify_without_a_secret_configured_rejects_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STUDIO_SESSION_SECRET", raising=False)
    get_settings.cache_clear()
    try:
        # No token can have been issued, so every candidate is simply invalid.
        # It must not raise on the authentication path.
        assert studio_session.verify("docie_sess_abc.def") is None
    finally:
        get_settings.cache_clear()


def test_minting_without_a_secret_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STUDIO_SESSION_SECRET", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(studio_session.StudioSessionUnavailableError):
            studio_session.mint("amine")
    finally:
        get_settings.cache_clear()


def test_malformed_tokens_are_rejected(signing_secret: None) -> None:
    for candidate in (
        "",
        "not-a-token",
        "docie_sess_",
        "docie_sess_onlypayload",
        "docie_sess_.sig",
        "docie_sess_payload.",
        "docie_sess_!!!.!!!",
    ):
        assert studio_session.verify(candidate) is None


def test_a_session_token_authenticates_as_the_company_login(signing_secret: None) -> None:
    manager = _manager()
    token = studio_session.mint("amine")

    context = manager.authenticate(token)

    assert context.authenticated
    # tenant = the company login, so usage is attributable to the person.
    assert context.tenant_id == "amine"


def test_an_invalid_session_token_401s_instead_of_falling_through(signing_secret: None) -> None:
    # A token carrying the session prefix must never be retried against the
    # operator keys: it is a bad credential, not a candidate key.
    manager = _manager(api_keys={"docie_sess_forged.sig": "sneaky"})

    with pytest.raises(HTTPException) as exc:
        manager.authenticate("docie_sess_forged.sig")
    assert exc.value.status_code == 401


def test_operator_keys_are_unaffected(signing_secret: None) -> None:
    manager = _manager(api_keys={"env-key": "acme"})

    context = manager.authenticate("env-key")

    assert context.authenticated
    assert context.tenant_id == "acme"


def test_ttl_never_outlives_the_company_token() -> None:
    # The exchange's whole premise is that company sign-in authorizes the
    # Studio, so Studio access must end no later than the session that proved
    # it. A company token with 15 minutes left yields a 15-minute session.
    assert studio_session.ttl_for(900) == 900
    assert studio_session.ttl_for(1800) == 1800


def test_ttl_is_capped_even_when_the_company_token_lives_longer() -> None:
    assert studio_session.ttl_for(7200) == studio_session.DEFAULT_TTL_SECONDS


def test_ttl_has_a_floor_so_a_token_is_never_dead_on_arrival() -> None:
    # A server reporting 0 (or a negative) must not mint something the browser
    # immediately has to re-exchange in a loop.
    assert studio_session.ttl_for(0) == studio_session.MIN_TTL_SECONDS
    assert studio_session.ttl_for(-30) == studio_session.MIN_TTL_SECONDS
    assert studio_session.ttl_for(5) == studio_session.MIN_TTL_SECONDS


def test_ttl_falls_back_when_the_server_reports_nothing_usable() -> None:
    for candidate in (None, "1800", True, [], {}):
        assert studio_session.ttl_for(candidate) == studio_session.DEFAULT_TTL_SECONDS


def test_a_session_minted_from_a_short_company_token_expires_with_it(
    signing_secret: None,
) -> None:
    issued_at = time.time()
    token = studio_session.mint("amine", ttl_seconds=studio_session.ttl_for(900), now=issued_at)

    assert studio_session.verify(token, now=issued_at + 899) == "amine"
    assert studio_session.verify(token, now=issued_at + 901) is None
