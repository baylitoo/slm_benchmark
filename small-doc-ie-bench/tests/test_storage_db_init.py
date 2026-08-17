from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect as sa_inspect

from docie_bench.storage import db


def test_base_schema_precedes_foreign_key_migrations(
    tmp_path: Path, monkeypatch
) -> None:
    """Fresh databases must have FK parents before additive child migrations."""
    original = db.ensure_review_evidence_table
    observed = False

    def guarded_migration(engine):
        nonlocal observed
        observed = True
        assert sa_inspect(engine).has_table("review_task")
        return original(engine)

    monkeypatch.setattr(db, "ensure_review_evidence_table", guarded_migration)
    try:
        db.init_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    finally:
        db.dispose_engine()

    assert observed
