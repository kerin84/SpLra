#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "Python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from research.benchmark_suite import run_research_benchmarks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the research benchmark and robustness suite.")
    parser.add_argument("--write-json", type=Path, help="Optional output path for benchmark results.")
    args = parser.parse_args()

    results = run_research_benchmarks()
    rendered = json.dumps(results, indent=2, sort_keys=True)

    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
