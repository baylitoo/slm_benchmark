"""JSON-file agent registry — CENTRAL config, not node state (#562).

Same persistence philosophy as the serving supervisor: one small JSON file,
read FRESH on every operation (no caching — the API replica that serves a
request may not be the one that wrote the record) and written atomically via
temp-file + ``os.replace``. Writes are last-writer-wins, which is fine for the
single-operator Studio; a multi-writer control plane would move this to the
Postgres catalog like the model store did.

WHERE IT LIVES, decided explicitly. The dependency map (#528) listed this as
one of the ``serving_home()`` readers, and #562 asked whether agents should be
catalog-backed or accepted as per-node config. Neither: an agent is **not node
state at all**. Nothing outside this package reads the registry, and the only
caller is the api router — a serving node has no use for ``agents.json`` and
should not carry one.

It sat under ``DOCIE_SERVING_HOME`` because that was the one shared volume, not
because the data is node-scoped. ``DOCIE_AGENTS_PATH`` now names it directly,
defaulting to the old location so existing deployments are unaffected. The
point of the setting is that a fleet can give the api its own config path
without the serving home — a per-node concept — deciding where fleet-wide
agent definitions live.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from docie_bench.agents.spec import AgentSpec, utcnow_iso
from docie_bench.serving.paths import serving_home


class AgentRegistryError(RuntimeError):
    pass


class AgentNotFoundError(AgentRegistryError):
    pass


class AgentConflictError(AgentRegistryError):
    pass


def default_agents_path() -> Path:
    """Where the agent registry lives.

    ``DOCIE_AGENTS_PATH`` when set; otherwise the historical location under the
    serving home, so nothing moves for an existing deployment.
    """
    configured = os.environ.get("DOCIE_AGENTS_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return serving_home() / "agents.json"


class AgentRegistry:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else default_agents_path()

    def list(self) -> list[AgentSpec]:
        data = self._read()
        return [AgentSpec.model_validate(record) for _, record in sorted(data.items())]

    def get(self, name: str) -> AgentSpec:
        data = self._read()
        record = data.get(name)
        if record is None:
            raise AgentNotFoundError(f"agent {name!r} does not exist")
        return AgentSpec.model_validate(record)

    def create(self, spec: AgentSpec) -> AgentSpec:
        data = self._read()
        if spec.name in data:
            raise AgentConflictError(f"agent {spec.name!r} already exists")
        data[spec.name] = spec.model_dump(mode="json")
        self._write(data)
        return spec

    def update(self, name: str, patch: dict[str, object]) -> AgentSpec:
        """Apply a partial update; ``name``/``created_at`` are immutable."""
        data = self._read()
        record = data.get(name)
        if record is None:
            raise AgentNotFoundError(f"agent {name!r} does not exist")
        merged = {**record, **patch, "name": name, "created_at": record.get("created_at")}
        merged["updated_at"] = utcnow_iso()
        spec = AgentSpec.model_validate(merged)
        data[name] = spec.model_dump(mode="json")
        self._write(data)
        return spec

    def delete(self, name: str) -> None:
        data = self._read()
        if name not in data:
            raise AgentNotFoundError(f"agent {name!r} does not exist")
        del data[name]
        self._write(data)

    def _read(self) -> dict[str, dict[str, object]]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise AgentRegistryError(f"agents registry at {self._path} is unreadable: {exc}") from exc
        if not isinstance(raw, dict):
            raise AgentRegistryError(f"agents registry at {self._path} is not a JSON object")
        return raw

    def _write(self, data: dict[str, dict[str, object]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self._path.parent, prefix=".agents-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_name, self._path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
