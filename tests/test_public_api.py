from __future__ import annotations

import numpy as np

from api.public_api import (
    evaluate_input_output_model,
    fit_autoregressive_model,
    fit_input_output_model,
    forecast_autoregressive_model,
    prepare_dataset_split,
    select_input_output_lag,
)


def _build_io_dataset(n_samples: int = 120) -> np.ndarray:
    rng = np.random.default_rng(123)
    u = rng.normal(scale=0.4, size=n_samples)
    y = np.zeros(n_samples)
    for k in range(1, n_samples):
        y[k] = 0.78 * y[k - 1] + 0.18 * u[k - 1]
    return np.vstack([u, y])


def test_prepare_dataset_split_loads_named_dataset() -> None:
    split = prepare_dataset_split("dryer", test_ratio=0.2, standardize=False)
    assert split.path == "Data/dryer.dat"
    assert split.train.shape[1] == 2
    assert split.test.shape[1] == 2


def test_select_and_evaluate_input_output_model() -> None:
    w = _build_io_dataset()
    train = w[:, :80]
    full = w

    lag_result = select_input_output_lag(train, n_inputs=1, lag_candidates=[2, 3, 4, 5], x0=0)
    assert lag_result.best_lag in {2, 3, 4, 5}
    assert lag_result.metric.shape == (4,)

    model = fit_input_output_model(train, lag=lag_result.best_lag, n_inputs=1, x0=0)
    evaluation = evaluate_input_output_model(model, full, test_horizon=40)

    assert evaluation.predicted_outputs.shape == (1, 40)
    assert np.isfinite(evaluation.fit_percent)
    assert np.isfinite(evaluation.misfit)


def test_autoregressive_forecast_returns_requested_horizon() -> None:
    y = _build_io_dataset()[1:2, :]
    model = fit_autoregressive_model(y[:, :90], lag=4, tol=1e-3, delta=1e-3)
    forecast = forecast_autoregressive_model(model, history=y[:, :90], horizon=8)

    assert forecast.forecast.shape == (1, 8)
    assert np.all(np.isfinite(forecast.forecast))
