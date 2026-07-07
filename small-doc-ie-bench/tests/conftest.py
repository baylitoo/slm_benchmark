from __future__ import annotations

import os

import pytest

# The API now fails closed by default (AUTH_REQUIRED=true, B3). The existing
# suite exercises /v1 routes via TestClient without sending keys, so default the
# test session to auth-off; tests that assert auth behaviour opt back in by
# monkeypatching docie_bench.security.get_quota_manager. os.environ takes
# precedence over any .env, so this wins regardless of a local .env file.
os.environ.setdefault("AUTH_REQUIRED", "false")
os.environ.setdefault("API_KEYS", "")

# ModelCatalog lazily runs init_engine() (so CLI entrypoints work without an
# explicit init), which reads DATABASE_URL from settings. Force it empty so a
# developer's local .env pointing at the live compose Postgres can never be
# picked up mid-test: tests that need a database init one explicitly (sqlite).
os.environ.setdefault("DATABASE_URL", "")


@pytest.fixture(autouse=True)
def _reset_quota_cache():
    """Isolate per-process rate-limit/concurrency state between tests."""
    from docie_bench import security

    def _clear() -> None:
        cache_clear = getattr(security.get_quota_manager, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()

    _clear()
    yield
    _clear()
