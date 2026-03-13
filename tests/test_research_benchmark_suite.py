from __future__ import annotations

from research.benchmark_suite import run_research_benchmarks


def test_research_benchmark_suite_returns_expected_cases() -> None:
    results = run_research_benchmarks()

    assert "hair_dryer" in results
    assert "cstr" in results

    for key in ("noise_robustness", "sample_robustness", "lag_sensitivity"):
        assert key in results["hair_dryer"]
        assert key in results["cstr"]

    assert results["hair_dryer"]["noise_robustness"]["x_values"] == [0.0, 0.01, 0.05]
    assert results["cstr"]["sample_robustness"]["x_values"] == [0.3, 0.5, 0.7, 1.0]
