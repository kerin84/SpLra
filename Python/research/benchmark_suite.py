from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from api.public_api import (
    InputOutputModel,
    evaluate_input_output_model,
    fit_input_output_model,
)
from experiments.paper_experiments import ExperimentConfig
from io_layer.sparse_lra_io import load_named_dataset, split_train_test, standardize_matrix


@dataclass(frozen=True)
class RobustnessCurve:
    x_values: list[float]
    sparse_lra_fit_percent: list[float]

    def to_dict(self) -> dict[str, list[float]]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    sparse_lra_validation_fit_percent: float
    persistence_validation_fit_percent: float
    mean_validation_fit_percent: float
    improvement_over_best_baseline: float
    noise_robustness: RobustnessCurve
    sample_robustness: RobustnessCurve
    lag_sensitivity: RobustnessCurve

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sparse_lra_validation_fit_percent": self.sparse_lra_validation_fit_percent,
            "persistence_validation_fit_percent": self.persistence_validation_fit_percent,
            "mean_validation_fit_percent": self.mean_validation_fit_percent,
            "improvement_over_best_baseline": self.improvement_over_best_baseline,
            "noise_robustness": self.noise_robustness.to_dict(),
            "sample_robustness": self.sample_robustness.to_dict(),
            "lag_sensitivity": self.lag_sensitivity.to_dict(),
        }


_RESEARCH_CONFIGS: tuple[tuple[str, ExperimentConfig, bool], ...] = (
    ("hair_dryer", ExperimentConfig("dryer", lag=5, n_inputs=1, test_ratio=0.5, tol=1e-2, delta=1e-2), False),
    ("cstr", ExperimentConfig("cstr", lag=1, n_inputs=1, test_ratio=0.5, tol=1e-2, delta=1e-2), True),
)


def _fit_percent(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    rec = np.asarray(reconstructed, dtype=float)
    misfit = float(np.linalg.norm(ref - rec, ord="fro"))
    centered = ref - np.mean(ref, axis=1, keepdims=True)
    denom = float(np.linalg.norm(centered, ord="fro"))
    if denom == 0:
        return 0.0
    return float(100 * (1 - (misfit / denom)))


def _prepare_data(dataset: str, standardize_outputs: bool) -> np.ndarray:
    loaded = load_named_dataset(dataset)
    if not standardize_outputs:
        return loaded.data
    standardized = standardize_matrix(loaded.data[:, 1:], axis=0)
    return standardized.data


def _fit_sparse_lra(
    data: np.ndarray,
    config: ExperimentConfig,
) -> tuple[InputOutputModel, np.ndarray, np.ndarray]:
    split = split_train_test(data, test_ratio=config.test_ratio, axis=0)
    train = np.asarray(split.train, dtype=float)
    test = np.asarray(split.test, dtype=float)
    model = fit_input_output_model(
        train.T,
        lag=config.lag,
        n_inputs=config.n_inputs,
        tol=config.tol,
        delta=config.delta,
    )
    return model, train, test


def _baseline_persistence(train: np.ndarray, test: np.ndarray, n_inputs: int) -> float:
    outputs_train = train[:, n_inputs:]
    outputs_test = test[:, n_inputs:]
    last_value = outputs_train[-1:, :]
    predicted = np.repeat(last_value, outputs_test.shape[0], axis=0).T
    return _fit_percent(outputs_test.T, predicted)


def _baseline_mean(train: np.ndarray, test: np.ndarray, n_inputs: int) -> float:
    outputs_train = train[:, n_inputs:]
    outputs_test = test[:, n_inputs:]
    mean_value = np.mean(outputs_train, axis=0, keepdims=True)
    predicted = np.repeat(mean_value, outputs_test.shape[0], axis=0).T
    return _fit_percent(outputs_test.T, predicted)


def _noise_robustness(data: np.ndarray, config: ExperimentConfig, noise_levels: list[float]) -> RobustnessCurve:
    rng = np.random.default_rng(3021984)
    fit_values: list[float] = []
    scale = np.std(data, axis=0, keepdims=True)
    scale_safe = np.where(scale < 1e-12, 1.0, scale)

    for noise in noise_levels:
        noisy = data + noise * scale_safe * rng.normal(size=data.shape)
        model, train, test = _fit_sparse_lra(noisy, config)
        evaluation = evaluate_input_output_model(model, noisy.T, test_horizon=test.shape[0])
        fit_values.append(evaluation.fit_percent)

    return RobustnessCurve(x_values=noise_levels, sparse_lra_fit_percent=fit_values)


def _sample_robustness(data: np.ndarray, config: ExperimentConfig, fractions: list[float]) -> RobustnessCurve:
    split = split_train_test(data, test_ratio=config.test_ratio, axis=0)
    full_train = np.asarray(split.train, dtype=float)
    test = np.asarray(split.test, dtype=float)
    fit_values: list[float] = []

    for fraction in fractions:
        n_train = max(config.lag + 2, int(round(full_train.shape[0] * fraction)))
        subset_train = full_train[:n_train, :]
        combined = np.vstack([subset_train, test])
        model = fit_input_output_model(
            subset_train.T,
            lag=config.lag,
            n_inputs=config.n_inputs,
            tol=config.tol,
            delta=config.delta,
        )
        evaluation = evaluate_input_output_model(model, combined.T, test_horizon=test.shape[0])
        fit_values.append(evaluation.fit_percent)

    return RobustnessCurve(x_values=fractions, sparse_lra_fit_percent=fit_values)


def _lag_sensitivity(data: np.ndarray, config: ExperimentConfig, lag_values: list[int]) -> RobustnessCurve:
    split = split_train_test(data, test_ratio=config.test_ratio, axis=0)
    train = np.asarray(split.train, dtype=float)
    test = np.asarray(split.test, dtype=float)
    combined = np.vstack([train, test])
    fit_values: list[float] = []

    for lag in lag_values:
        model = fit_input_output_model(
            train.T,
            lag=lag,
            n_inputs=config.n_inputs,
            tol=config.tol,
            delta=config.delta,
        )
        evaluation = evaluate_input_output_model(model, combined.T, test_horizon=test.shape[0])
        fit_values.append(evaluation.fit_percent)

    return RobustnessCurve(x_values=[float(lag) for lag in lag_values], sparse_lra_fit_percent=fit_values)


def run_research_benchmarks() -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}

    for name, config, standardize_outputs in _RESEARCH_CONFIGS:
        data = _prepare_data(config.dataset, standardize_outputs=standardize_outputs)
        model, train, test = _fit_sparse_lra(data, config)
        evaluation = evaluate_input_output_model(model, data.T, test_horizon=test.shape[0])
        persistence_fit = _baseline_persistence(train, test, config.n_inputs)
        mean_fit = _baseline_mean(train, test, config.n_inputs)
        best_baseline = max(persistence_fit, mean_fit)

        if config.lag <= 1:
            lag_candidates = [1, 2, 3]
        else:
            lag_candidates = [config.lag - 1, config.lag, config.lag + 1]

        benchmark = BenchmarkResult(
            name=name,
            sparse_lra_validation_fit_percent=evaluation.fit_percent,
            persistence_validation_fit_percent=persistence_fit,
            mean_validation_fit_percent=mean_fit,
            improvement_over_best_baseline=evaluation.fit_percent - best_baseline,
            noise_robustness=_noise_robustness(data, config, noise_levels=[0.0, 0.01, 0.05]),
            sample_robustness=_sample_robustness(data, config, fractions=[0.3, 0.5, 0.7, 1.0]),
            lag_sensitivity=_lag_sensitivity(data, config, lag_values=lag_candidates),
        )
        results[name] = benchmark.to_dict()

    return results
