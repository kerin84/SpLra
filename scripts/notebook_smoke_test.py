#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MARKER = "# Reproducibility setup (local and Colab-friendly)"
LOAD_PATTERNS = (
    re.compile(r"\bnp\.loadtxt\("),
    re.compile(r"\bpd\.read_csv\("),
)


def _iter_code_cells(notebook: dict) -> list[list[str]]:
    return [cell.get("source", []) for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]


def _sanitize_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return line
    if stripped.startswith(("%", "!", "cd ")):
        return None
    return line


def _exec_source(lines: list[str], env: dict, label: str) -> None:
    sanitized = []
    for line in lines:
        out = _sanitize_line(line)
        if out is not None:
            sanitized.append(out)
    source = "".join(sanitized).strip()
    if not source:
        return
    try:
        exec(compile(source, label, "exec"), env)
    except Exception as exc:
        raise RuntimeError(f"Execution failed in {label}: {exc}") from exc


def _find_setup_cell(code_cells: list[list[str]]) -> list[str]:
    for src in code_cells:
        if MARKER in "".join(src):
            return src
    raise RuntimeError("Reproducibility setup cell not found.")


def _find_first_data_load_line(code_cells: list[list[str]]) -> str | None:
    for src in code_cells:
        for line in src:
            if any(p.search(line) for p in LOAD_PATTERNS):
                return line
    return None


def smoke_test_notebook(path: Path, verbose: bool = False) -> tuple[bool, str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = _iter_code_cells(notebook)

    env: dict = {
        "__name__": "__main__",
        "np": np,
        "pd": pd,
        "Path": Path,
        "sys": sys,
    }

    try:
        setup = _find_setup_cell(code_cells)
        _exec_source(setup, env, f"{path.name}:setup")

        load_line = _find_first_data_load_line(code_cells)
        if load_line is not None:
            _exec_source([load_line], env, f"{path.name}:data_load")
            return True, "setup+data_load"

        # No explicit data-loading line in this notebook: setup smoke test only.
        return True, "setup_only"
    except Exception as exc:
        if verbose:
            return False, f"{exc}"
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test notebook reproducibility setup and data loading.")
    parser.add_argument("--notebooks-dir", default="Notebooks", help="Directory containing .ipynb notebooks.")
    parser.add_argument("--verbose", action="store_true", help="Show detailed errors.")
    args = parser.parse_args()

    notebooks_dir = Path(args.notebooks_dir)
    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {notebooks_dir}")
        return 1

    failures = []
    for nb in notebooks:
        ok, mode = smoke_test_notebook(nb, verbose=args.verbose)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {nb} ({mode})")
        if not ok:
            failures.append((nb, mode))

    if failures:
        print("\nSmoke test failures:")
        for nb, msg in failures:
            print(f"- {nb}: {msg}")
        return 1

    print("\nNotebook smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
