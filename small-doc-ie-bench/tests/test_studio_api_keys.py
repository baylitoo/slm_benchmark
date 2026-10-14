from __future__ import annotations

from pathlib import Path

import pytest

from docie_bench.security import TenantQuotaManager
from docie_bench.settings import get_settings
from docie_bench.storage.db import dispose_engine, init_engine
from docie_bench.studio import key_crypto
from docie_bench.studio.api_key_store import (
    create_key,
    list_keys,
    lookup_by_secret,
    reveal_key,
    revoke_key,
    split_key,
    usage_by_key,
)
from docie_bench.studio.key_crypto import KeyEncryptionUnavailableError
from docie_bench.studio.usage_store import record_usage


@pytest.fixture
def studio_database(tmp_path: Path):
    """Same shape as test_extraction_results' fixture: a throwaway sqlite file
    so the store's own migrations run exactly as they do against PostgreSQL."""
    init_engine(f"sqlite:///{tmp_path / 'api_keys.db'}")
    yield
    dispose_engine()


@pytest.fixture
def encryption_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY_ENCRYPTION_KEY", "test-encryption-key")
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


def test_split_key_keeps_a_secret_that_contains_underscores() -> None:
    # token_urlsafe emits "-" and "_", so a naive 3-way split truncates
    # roughly half of all generated keys at the first underscore.
    assert split_key("docie_abc123_sec_ret_with_underscores") == (
        "abc123",
        "sec_ret_with_underscores",
    )
    assert split_key("docie_abc123_plain") == ("abc123", "plain")


def test_split_key_rejects_other_shapes() -> None:
    assert split_key("docie_onlyprefix") is None
    assert split_key("other_abc_secret") is None
    assert split_key("docie__secret") is None
    assert split_key("") is None


def test_encrypt_roundtrip(encryption_key: None) -> None:
    sealed = key_crypto.encrypt_secret("s3cr3t")
    assert "s3cr3t" not in sealed
    assert key_crypto.decrypt_secret(sealed) == "s3cr3t"


def test_encrypt_uses_a_fresh_nonce_per_call(encryption_key: None) -> None:
    # Nonce reuse under one AES-GCM key breaks confidentiality AND
    # authentication, so two encryptions of the same plaintext must differ.
    first = key_crypto.encrypt_secret("same")
    second = key_crypto.encrypt_secret("same")
    assert first != second
    assert key_crypto.decrypt_secret(first) == key_crypto.decrypt_secret(second) == "same"


def test_decrypt_under_a_different_key_is_an_honest_configuration_error(
    encryption_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = key_crypto.encrypt_secret("s3cr3t")
    monkeypatch.setenv("API_KEY_ENCRYPTION_KEY", "a-rotated-key")
    get_settings.cache_clear()
    with pytest.raises(KeyEncryptionUnavailableError, match="does not match"):
        key_crypto.decrypt_secret(sealed)


def test_creation_without_an_encryption_key_refuses_instead_of_storing_cleartext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("API_KEY_ENCRYPTION_KEY", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(KeyEncryptionUnavailableError):
            key_crypto.encrypt_secret("would-be-cleartext")
    finally:
        get_settings.cache_clear()


def test_env_keys_are_the_operator_tier() -> None:
    manager = _manager(api_keys={"env-key": "acme"})
    context = manager.authenticate("env-key")
    assert context.authenticated
    assert context.tenant_id == "acme"
    assert context.operator is True
    # No key row to attribute usage to.
    assert context.api_key_id is None


def test_issued_keys_authenticate_as_the_client_tier() -> None:
    manager = _manager(issued_key_lookup=lambda key: ("acme", "key-1"))
    context = manager.authenticate("docie_abc_secret")
    assert context.authenticated
    assert context.tenant_id == "acme"
    assert context.api_key_id == "key-1"
    # The whole point of the split: a client key cannot mint more keys.
    assert context.operator is False


def test_env_keys_are_checked_before_issued_keys() -> None:
    # An env key must never be shadowed by a database row claiming the same
    # secret, or a compromised catalog could silently re-tenant operators.
    manager = _manager(
        api_keys={"shared": "env-tenant"},
        issued_key_lookup=lambda key: ("db-tenant", "key-1"),
    )
    context = manager.authenticate("shared")
    assert context.tenant_id == "env-tenant"
    assert context.operator is True


def test_an_unknown_key_still_401s_with_a_lookup_configured() -> None:
    from fastapi import HTTPException

    manager = _manager(issued_key_lookup=lambda key: None)
    with pytest.raises(HTTPException) as exc:
        manager.authenticate("docie_abc_nope")
    assert exc.value.status_code == 401


@pytest.mark.usefixtures("studio_database")
def test_create_returns_the_key_once_then_only_a_masked_view(encryption_key: None) -> None:
    view, full_key = create_key(label="CI pipeline", tenant_id="acme")

    assert full_key.startswith("docie_")
    assert view["label"] == "CI pipeline"
    assert view["revoked"] is False
    # The masked view must never carry the secret, in any field.
    assert full_key not in str(view)
    listed = list_keys()
    assert [row["id"] for row in listed] == [view["id"]]
    assert full_key not in str(listed)
    assert listed[0]["prefix"] == view["prefix"]


@pytest.mark.usefixtures("studio_database")
def test_a_live_key_authenticates_and_a_revoked_one_stops_immediately(
    encryption_key: None,
) -> None:
    view, full_key = create_key(label="client", tenant_id="acme")

    assert lookup_by_secret(full_key) == ("acme", view["id"])

    revoke_key(view["id"])
    # No cache anywhere: the very next request must fail, on every replica.
    assert lookup_by_secret(full_key) is None


@pytest.mark.usefixtures("studio_database")
def test_a_wrong_secret_on_a_real_prefix_does_not_authenticate(
    encryption_key: None,
) -> None:
    _, full_key = create_key(label="client", tenant_id="acme")
    prefix, _secret = split_key(full_key)  # type: ignore[misc]

    assert lookup_by_secret(f"docie_{prefix}_not-the-secret") is None


@pytest.mark.usefixtures("studio_database")
def test_revoke_is_idempotent_and_keeps_the_original_timestamp(
    encryption_key: None,
) -> None:
    view, _ = create_key(label="client", tenant_id="acme")

    first = revoke_key(view["id"])
    second = revoke_key(view["id"])

    assert first is not None
    assert second is not None
    assert first["revoked"] is True
    assert second["revoked_at"] == first["revoked_at"]


@pytest.mark.usefixtures("studio_database")
def test_reveal_returns_the_same_key_that_was_issued(encryption_key: None) -> None:
    view, full_key = create_key(label="client", tenant_id="acme")

    assert reveal_key(view["id"]) == full_key
    assert reveal_key("not-a-key-id") is None


@pytest.mark.usefixtures("studio_database")
def test_usage_attributes_to_the_key_and_unused_keys_still_appear(
    encryption_key: None,
) -> None:
    used, _ = create_key(label="busy", tenant_id="acme")
    idle, _ = create_key(label="idle", tenant_id="acme")

    record_usage(
        deployment="lfm2.5-2.6b",
        surface="chat",
        tenant_id="acme",
        latency_ms=120,
        prompt_tokens=10,
        completion_tokens=5,
        api_key_id=used["id"],
    )
    record_usage(
        deployment="lfm2.5-2.6b",
        surface="chat",
        tenant_id="acme",
        latency_ms=80,
        status="error",
        api_key_id=used["id"],
    )
    # An env-key request: attributable to the tenant, to no key.
    record_usage(
        deployment="lfm2.5-2.6b", surface="chat", tenant_id="acme", latency_ms=50
    )

    rows = {row["id"]: row for row in usage_by_key(window_hours=24)}

    assert rows[used["id"]]["requests"] == 2
    assert rows[used["id"]]["prompt_tokens"] == 10
    assert rows[used["id"]]["completion_tokens"] == 5
    assert rows[used["id"]]["errors"] == 1
    assert rows[used["id"]]["avg_latency_ms"] == 100
    # A key with no traffic must be listed with zeros, not omitted -- otherwise
    # the view cannot answer "is this key still being used".
    assert rows[idle["id"]]["requests"] == 0
    assert rows[idle["id"]]["avg_latency_ms"] is None
