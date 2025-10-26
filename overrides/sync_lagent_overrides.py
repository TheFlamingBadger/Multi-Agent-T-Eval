#!/usr/bin/env python3

"""
Copy the local lagent overrides back into the nested lagent package.

This repository keeps customised versions of lagent's `openai.py` and
`huggingface.py` inside `setup/` so they can be tracked outside the nested
git-ignored checkout. Running this script copies those files into the actual
`lagent/lagent/llms/` package that the application imports at runtime.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def copy_override(src: Path, dest: Path, repo_root: Path) -> None:
    """Copy a single override file into place."""
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)

    try:
        src_display = src.relative_to(repo_root)
        dest_display = dest.relative_to(repo_root)
    except ValueError:
        # Fallback in the unlikely event the files sit outside the repo tree.
        src_display = src
        dest_display = dest

    print(f"Copied {src_display} -> {dest_display}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    targets = {
        script_dir / "openai.py": repo_root / "lagent/lagent/llms/openai.py",
        script_dir / "huggingface.py": repo_root / "lagent/lagent/llms/huggingface.py",
    }

    for src, dest in targets.items():
        copy_override(src, dest, repo_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
