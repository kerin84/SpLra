#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()
PYTHON_DIR = PROJECT_ROOT / "Python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from experiments.paper_experiments import run_paper_experiments  # noqa: E402


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare_metrics(
    current: dict[str, dict[str, object]],
    expected: dict[str, dict[str, object]],
    tolerance: float,
) -> list[str]:
    failures: list[str] = []

    for experiment_name, expected_metrics in expected.items():
        current_metrics = current.get(experiment_name)
        if current_metrics is None:
            failures.append(f"Missing experiment: {experiment_name}")
            continue

        for key, expected_value in expected_metrics.items():
            if key not in current_metrics:
                failures.append(f"{experiment_name}: missing metric '{key}'")
                continue

            current_value = current_metrics[key]
            if _is_number(expected_value) and _is_number(current_value):
                if not math.isclose(
                    float(current_value),
                    float(expected_value),
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                ):
                    failures.append(
                        f"{experiment_name}.{key}: expected {expected_value}, got {current_value}"
                    )
            elif current_value != expected_value:
                failures.append(
                    f"{experiment_name}.{key}: expected {expected_value!r}, got {current_value!r}"
                )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run canonical paper experiments outside notebooks.")
    parser.add_argument(
        "--check",
        type=Path,
        help="Compare current metrics against a baseline JSON file.",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        help="Write the current experiment metrics to the given JSON path.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute and relative tolerance used with --check.",
    )
    args = parser.parse_args()

    results = run_paper_experiments()

    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.check is not None:
        expected = _load_json(args.check)
        failures = _compare_metrics(results, expected, tolerance=args.tolerance)
        if failures:
            print("Paper experiment check failed:")
            for failure in failures:
                print(f"- {failure}")
            return 1
        print(f"Paper experiment check passed against {args.check}")
        return 0

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
