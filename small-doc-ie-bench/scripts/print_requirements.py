"""Print the project's dependency list (base + requested extras), one per line.

Docker build helper: lets the image install third-party dependencies from
pyproject.toml BEFORE the source tree is copied, so code edits never
invalidate the (expensive, torch-sized) dependency layer. Stdlib only.

Usage: python scripts/print_requirements.py "ocr,encoders"
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def main() -> None:
    extras = [e.strip() for e in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if e.strip()]
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    requirements: list[str] = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    for extra in extras:
        if extra not in optional:
            raise SystemExit(f"unknown extra {extra!r} (available: {', '.join(sorted(optional))})")
        requirements.extend(optional[extra])
    print("\n".join(requirements))


if __name__ == "__main__":
    main()
