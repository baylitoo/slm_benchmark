"""Download a GGUF model from Hugging Face into ./models.

Example:
  python scripts/download_gguf.py --repo TheBloke/Some-GGUF --filename model.Q4_K_M.gguf

This script is intentionally thin because model choice changes frequently.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("pip install huggingface_hub first") from exc
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=args.repo, filename=args.filename, local_dir=out_dir)
    print(path)


if __name__ == "__main__":
    main()
