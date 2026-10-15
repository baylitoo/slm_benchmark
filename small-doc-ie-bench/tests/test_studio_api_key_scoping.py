"""Who may manage which API keys.

Everyone signed in with a company account may issue and revoke keys -- that is
part of what the Studio is for -- but only their own. These tests are the guard
on "only their own", which is the half that is easy to lose in a refactor.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docie_bench.security import get_quota_manager
from docie_bench.settings import get_settings
from docie_bench.storage.db import dispose_engine, init_engine
from docie_bench.studio_session import mint

ALICE = "alice@cynov.com"
BOB = "bob@cynov.com"
OPERATOR_KEY = "operator-secret"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("API_KEYS", f"{OPERATOR_KEY}:ops")
    monkeypatch.setenv("STUDIO_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("API_KEY_ENCRYPTION_KEY", "test-encryption-key")
    get_settings.cache_clear()
    get_quota_manager.cache_clear()
    init_engine(f"sqlite:///{tmp_path / 'scoping.db'}")
    from docie_bench.api import app

    with TestClient(app) as test_client:
        yield test_client
    dispose_engine()
    get_settings.cache_clear()
    get_quota_manager.cache_clear()


def _as(login: str) -> dict[str, str]:
    return {"X-API-Key": mint(login)}


def _operator() -> dict[str, str]:
    return {"X-API-Key": OPERATOR_KEY}


def _create(client: TestClient, headers: dict[str, str], label: str, **body: str) -> dict:
    resp = client.post("/v1/studio/api-keys", headers=headers, json={"label": label, **body})
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_a_signed_in_user_can_issue_a_key(client: TestClient) -> None:
    created = _create(client, _as(ALICE), "alice's CI")

    assert created["key"].startswith("docie_")
    assert created["tenant_id"] == ALICE


def test_the_owner_is_the_verified_login_not_what_the_body_claims(
    client: TestClient,
) -> None:
    # The request body is not evidence of anything. Without this override any
    # signed-in user could mint a key owned by somebody else.
    created = _create(client, _as(ALICE), "sneaky", tenant_id=BOB)

    assert created["tenant_id"] == ALICE


def test_an_operator_may_still_provision_for_another_tenant(client: TestClient) -> None:
    created = _create(client, _operator(), "provisioned", tenant_id="acme")

    assert created["tenant_id"] == "acme"


def test_a_user_sees_only_their_own_keys(client: TestClient) -> None:
    mine = _create(client, _as(ALICE), "mine")
    theirs = _create(client, _as(BOB), "theirs")

    listed = client.get("/v1/studio/api-keys", headers=_as(ALICE)).json()

    ids = [row["id"] for row in listed["keys"]]
    assert ids == [mine["id"]]
    assert theirs["id"] not in ids
    # So the UI can say "your keys" rather than implying the whole fleet.
    assert listed["scoped_to"] == ALICE


def test_an_operator_sees_every_key(client: TestClient) -> None:
    mine = _create(client, _as(ALICE), "mine")
    theirs = _create(client, _as(BOB), "theirs")

    listed = client.get("/v1/studio/api-keys", headers=_operator()).json()

    ids = {row["id"] for row in listed["keys"]}
    assert {mine["id"], theirs["id"]} <= ids
    assert listed["scoped_to"] is None


def test_revoking_someone_elses_key_is_a_404_not_a_403(client: TestClient) -> None:
    theirs = _create(client, _as(BOB), "theirs")

    resp = client.delete(f"/v1/studio/api-keys/{theirs['id']}", headers=_as(ALICE))

    # 404, deliberately: a 403 would confirm the key id exists, which is what a
    # caller probing for other people's keys wants to learn.
    assert resp.status_code == 404
    # And it must not have been revoked.
    still = client.get("/v1/studio/api-keys", headers=_as(BOB)).json()["keys"]
    assert still[0]["revoked"] is False


def test_revealing_someone_elses_key_is_a_404(client: TestClient) -> None:
    theirs = _create(client, _as(BOB), "theirs")

    resp = client.post(f"/v1/studio/api-keys/{theirs['id']}/reveal", headers=_as(ALICE))

    assert resp.status_code == 404


def test_a_user_can_revoke_and_reveal_their_own(client: TestClient) -> None:
    mine = _create(client, _as(ALICE), "mine")

    revealed = client.post(f"/v1/studio/api-keys/{mine['id']}/reveal", headers=_as(ALICE))
    assert revealed.status_code == 200
    assert revealed.json()["key"] == mine["key"]

    revoked = client.delete(f"/v1/studio/api-keys/{mine['id']}", headers=_as(ALICE))
    assert revoked.status_code == 200
    assert revoked.json()["revoked"] is True


def test_an_issued_key_cannot_manage_keys_at_all(client: TestClient) -> None:
    created = _create(client, _as(ALICE), "a client key")

    # Authenticating WITH the issued key: it may call the serving surfaces, but
    # if it could mint a successor, revoking a leaked key would be pointless.
    resp = client.get("/v1/studio/api-keys", headers={"X-API-Key": created["key"]})

    assert resp.status_code == 403


def test_usage_is_scoped_the_same_way(client: TestClient) -> None:
    mine = _create(client, _as(ALICE), "mine")
    _create(client, _as(BOB), "theirs")

    rows = client.get("/v1/studio/api-keys/usage?window=24h", headers=_as(ALICE)).json()

    assert [row["id"] for row in rows["keys"]] == [mine["id"]]
