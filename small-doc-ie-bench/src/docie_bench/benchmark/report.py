from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from jinja2 import Template

from docie_bench.benchmark.reproducibility import atomic_write_json, atomic_write_text

HTML_TEMPLATE = Template(
    """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Small Document IE Benchmark Report</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem; max-width: 900px; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
    th, td { border: 1px solid #ddd; padding: .45rem; text-align: left; }
    th { background: #f6f6f6; }
    .ok { color: #006400; font-weight: 600; }
    .bad { color: #8b0000; font-weight: 600; }
    code { background: #f4f4f4; padding: .15rem .25rem; border-radius: 4px; }
    .cpu-section { margin-top: 2rem; }
    .cpu-meta { font-size: .85rem; color: #555; margin: .25rem 0 .75rem; }
  </style>
</head>
<body>
  <h1>Small Document IE Benchmark Report</h1>
  <p>Run directory: <code>{{ run_dir }}</code></p>
  <h2>Summary</h2>
  <table>
    <thead><tr>
      <th>Model profile</th><th>Input path</th><th>Docs</th><th>Concurrency</th>
      <th>Wall time</th><th>Throughput (docs/min)</th>
      <th>Valid JSON</th><th>Field accuracy</th>
      <th>Row F1</th>
      <th>Evidence coverage</th><th>Row evidence coverage</th><th>Hallucination rate</th>
      <th>Judge faithfulness</th><th>Judge completeness</th>
      <th>Avg ms</th><th>p50 ms</th><th>p95 ms</th>
    </tr></thead>
    <tbody>
    {% for row in summary %}
      <tr>
        <td>{{ row.model_profile }}</td>
        <td>{{ row.ingestion_path }}</td>
        <td>{{ row.docs }}</td>
        <td>{{ row.concurrency }}</td>
        <td>{{ row.wall_seconds }}s</td>
        <td>{{ '%.2f'|format(row.throughput_docs_per_min) if row.throughput_docs_per_min is not none else '—' }}</td>
        <td>{{ '%.1f'|format(row.valid_rate * 100) }}%</td>
        <td>{{ '%.1f'|format(row.field_accuracy * 100 if row.field_accuracy is not none else 0) }}%</td>
        <td>
          {{ '%.1f%%'|format(row.get('row_f1') * 100)
             if row.get('row_f1') is not none else 'N/A' }}
        </td>
        <td>
          {{ '%.1f'|format(row.evidence_coverage * 100 if row.evidence_coverage is not none else 0) }}%
        </td>
        <td>
          {{ '%.1f%%'|format(row.get('evidence_row_coverage') * 100)
             if row.get('evidence_row_coverage') is not none else 'N/A' }}
        </td>
        <td>
          {{ '%.1f'|format(row.hallucination_rate * 100 if row.hallucination_rate is not none else 0) }}%
        </td>
        <td>
          {{ '%.1f%%'|format(row.get('judge_faithfulness') * 100)
             if row.get('judge_faithfulness') is not none else 'N/A' }}
        </td>
        <td>
          {{ '%.1f%%'|format(row.get('judge_completeness') * 100)
             if row.get('judge_completeness') is not none else 'N/A' }}
        </td>
        <td>{{ '%.0f'|format(row.avg_latency_ms) }}</td>
        <td>{{ row.p50_latency_ms }}</td>
        <td>{{ row.p95_latency_ms }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Table Extraction</h2>
  <table>
    <thead><tr>
      <th>Document</th><th>Model profile</th><th>Table</th>
      <th>Rows correct</th><th>Rows expected</th><th>Rows predicted</th>
    </tr></thead>
    <tbody>
    {% for row in rows if row.score and row.score.tables %}
      {% for table in row.score.tables %}
      <tr>
        <td>{{ row.doc_id }}</td><td>{{ row.model_profile }}</td><td>{{ table.field }}</td>
        <td>{{ table.row_correct }}</td>
        <td>{{ table.row_expected }}</td>
        <td>{{ table.row_predicted }}</td>
      </tr>
      {% endfor %}
    {% else %}
      <tr><td colspan="6">No table ground truth in this run.</td></tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Arithmetic Validation Warnings</h2>
  <table>
    <thead><tr><th>Document</th><th>Model profile</th><th>Warnings</th></tr></thead>
    <tbody>
    {% for row in rows if row.validation and row.validation.warnings %}
      <tr>
        <td>{{ row.doc_id }}</td><td>{{ row.model_profile }}</td>
        <td class="bad">{{ row.validation.warnings | join('; ') }}</td>
      </tr>
    {% else %}
      <tr><td colspan="3" class="ok">No arithmetic validation warnings.</td></tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Ungrounded Fields</h2>
  <table>
    <thead><tr><th>Document</th><th>Model profile</th><th>Potential hallucinations</th></tr></thead>
    <tbody>
    {% for row in rows if row.score and row.score.ungrounded_fields %}
      <tr>
        <td>{{ row.doc_id }}</td>
        <td>{{ row.model_profile }}</td>
        <td class="bad">{{ row.score.ungrounded_fields | join(', ') }}</td>
      </tr>
    {% else %}
      <tr><td colspan="3" class="ok">All extracted fields are grounded.</td></tr>
    {% endfor %}
    </tbody>
  </table>

  <div class="cpu-section">
    <h2>CPU Usage During Benchmark</h2>
    {% if cpu_chart %}
      <p class="cpu-meta">
        Sampled every 1 s &nbsp;|&nbsp;
        Peak: <strong>{{ cpu_peak }}%</strong> &nbsp;|&nbsp;
        Average: <strong>{{ cpu_avg }}%</strong> &nbsp;|&nbsp;
        Duration: <strong>{{ cpu_duration }}s</strong>
        &nbsp;&mdash;&nbsp; <span style="color:#2563eb">&#9644; measured</span>
        &nbsp; <span style="color:#ef4444">&#8211;&#8211; avg</span>
      </p>
      {{ cpu_chart }}
    {% else %}
      <p><em>No CPU samples (psutil not installed).</em></p>
    {% endif %}
  </div>

  <h2>Artifacts</h2>
  <ul>
    <li><code>manifest.json</code></li>
    <li><code>task-events.jsonl</code></li>
    <li><code>predictions.jsonl</code></li>
    <li><code>metrics.json</code></li>
  </ul>
  {% if reproducibility %}
  <h2>Reproducibility</h2>
  <table>
    <tbody>
      <tr>
        <th>Input fingerprint</th>
        <td><code>{{ reproducibility.input_fingerprint }}</code></td>
      </tr>
      <tr><th>Tasks</th><td>{{ reproducibility.task_count }}</td></tr>
      <tr><th>Resumed</th><td>{{ reproducibility.resumed }}</td></tr>
      <tr>
        <th>Skipped / executed</th>
        <td>{{ reproducibility.tasks_skipped }} / {{ reproducibility.tasks_executed }}</td>
      </tr>
      <tr><th>Manifest</th><td><code>{{ reproducibility.manifest_path }}</code></td></tr>
      <tr><th>Warnings</th><td>{{ reproducibility.warnings | join('; ') }}</td></tr>
    </tbody>
  </table>
  {% endif %}
  {% if manifest_json %}
  <details>
    <summary><strong>Immutable run manifest</strong></summary>
    <pre>{{ manifest_json }}</pre>
  </details>
  {% endif %}
</body>
</html>
"""
)


def _cpu_chart_svg(samples: list[tuple[float, float]]) -> str:
    if not samples:
        return ""

    W, H = 780, 200
    LEFT, RIGHT, TOP, BOTTOM = 42, 16, 16, 28
    CW = W - LEFT - RIGHT
    CH = H - TOP - BOTTOM

    times = [s[0] for s in samples]
    cpus = [s[1] for s in samples]
    t_max = max(times) or 1.0

    def px(t: float) -> float:
        return LEFT + (t / t_max) * CW

    def py(c: float) -> float:
        return TOP + CH - (c / 100.0) * CH

    points = " ".join(f"{px(t):.1f},{py(c):.1f}" for t, c in samples)
    avg = sum(cpus) / len(cpus)
    avg_y = py(avg)

    grid_lines = ""
    for pct in (0, 25, 50, 75, 100):
        gy = py(pct)
        grid_lines += (
            f'<line x1="{LEFT}" y1="{gy:.1f}" x2="{W - RIGHT}" y2="{gy:.1f}"'
            f' stroke="#e5e7eb" stroke-dasharray="3"/>'
            f'<text x="{LEFT - 4}" y="{gy + 4:.0f}" font-size="10"'
            f' fill="#9ca3af" text-anchor="end">{pct}%</text>'
        )

    x_ticks = ""
    n_ticks = min(6, len(samples))
    for i in range(n_ticks + 1):
        t = (i / n_ticks) * t_max
        tx = px(t)
        x_ticks += (
            f'<text x="{tx:.1f}" y="{H - 6}" font-size="10"'
            f' fill="#9ca3af" text-anchor="middle">{t:.0f}s</text>'
        )

    return (
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}"'
        f' xmlns="http://www.w3.org/2000/svg" style="display:block;max-width:100%">'
        f'<rect width="{W}" height="{H}" fill="#f9fafb" rx="4"/>'
        f"{grid_lines}"
        f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + CH}" stroke="#d1d5db"/>'
        f'<line x1="{LEFT}" y1="{TOP + CH}" x2="{W - RIGHT}" y2="{TOP + CH}" stroke="#d1d5db"/>'
        f'<polyline points="{points}" fill="none" stroke="#2563eb"'
        f' stroke-width="2" stroke-linejoin="round"/>'
        f'<line x1="{LEFT}" y1="{avg_y:.1f}" x2="{W - RIGHT}" y2="{avg_y:.1f}"'
        f' stroke="#ef4444" stroke-dasharray="6,3" stroke-width="1.5"/>'
        f'<text x="{W - RIGHT - 2}" y="{avg_y - 5:.0f}" font-size="10"'
        f' fill="#ef4444" text-anchor="end">avg {avg:.0f}%</text>'
        f"{x_ticks}"
        f"</svg>"
    )


def write_report(run_dir: Path, metrics: dict[str, Any]) -> Path:
    cpu_samples: list[tuple[float, float]] = [
        (s[0], s[1]) for s in metrics.get("cpu_samples", [])
    ]
    cpus = [s[1] for s in cpu_samples]
    cpu_chart = _cpu_chart_svg(cpu_samples)
    manifest_path = run_dir / "manifest.json"
    manifest_json = ""
    if manifest_path.exists():
        manifest_json = html.escape(
            json.dumps(json.loads(manifest_path.read_text(encoding="utf-8")), indent=2)
        )

    path = run_dir / "report.html"
    atomic_write_text(
        path,
        HTML_TEMPLATE.render(
            run_dir=str(run_dir),
            summary=metrics.get("summary", []),
            rows=metrics.get("rows", []),
            reproducibility=metrics.get("reproducibility"),
            manifest_json=manifest_json,
            cpu_chart=cpu_chart,
            cpu_peak=f"{max(cpus):.0f}" if cpus else "—",
            cpu_avg=f"{sum(cpus) / len(cpus):.0f}" if cpus else "—",
            cpu_duration=f"{max(s[0] for s in cpu_samples):.0f}" if cpu_samples else "—",
        ),
    )
    atomic_write_json(run_dir / "metrics.pretty.json", metrics, indent=2)
    return path
