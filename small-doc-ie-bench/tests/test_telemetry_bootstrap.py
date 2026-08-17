from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_missing_prometheus_multiprocess_dir_is_created(tmp_path: Path) -> None:
    multiproc_dir = tmp_path / "missing" / "prometheus"
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PROMETHEUS_MULTIPROC_DIR"] = str(multiproc_dir)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(project_root / "src"), env.get("PYTHONPATH", "")]
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from docie_bench.telemetry import AGENT_REQUESTS, generate_metrics; "
                "AGENT_REQUESTS.labels('ocr-test', 'ocr', 'ok').inc(); "
                "assert b'docie_agent_requests_total' in generate_metrics()"
            ),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert multiproc_dir.is_dir()
    assert any(multiproc_dir.glob("counter_*.db"))
