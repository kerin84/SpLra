from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import control as cnt
import numpy as np
import scipy.linalg as la

from core.sparse_lra_core import (
    SparseLraAutSysIdResult,
    SparseLraSysIdResult,
    sparse_lra_aut_sysid,
    sparse_lra_sysid,
)
from io_layer.sparse_lra_io import (
    StandardizationParams,
    load_named_dataset,
    load_matrix,
    resolve_data_path,
    split_train_test,
    standardize_matrix,
)


@dataclass(frozen=True)
class DatasetSplit:
    source: str
    path: str
    train: np.ndarray
    test: np.ndarray
    standardized: bool
    standardization_params: StandardizationParams | None


@dataclass(frozen=True)
class InputOutputModel:
    lag: int
    n_inputs: int
    tol: float
    delta: float
    result: SparseLraSysIdResult


@dataclass(frozen=True)
class ValidationMetrics:
    fit_percent: float
    misfit: float
    predicted_outputs: np.ndarray
    reference_outputs: np.ndarray


@dataclass(frozen=True)
class LagSelectionResult:
    best_lag: int
    best_metric: float
    lag_values: np.ndarray
    metric: np.ndarray


@dataclass(frozen=True)
class AutoregressiveModel:
    lag: int
    tol: float
    delta: float
    result: SparseLraAutSysIdResult


@dataclass(frozen=True)
class AutoregressiveForecast:
    history: np.ndarray
    horizon: int
    forecast: np.ndarray


def _repo_relative_path(path: Path) -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _ensure_2d_rows(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _fit_percent(reference: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=float)
    rec = np.asarray(reconstructed, dtype=float)
    misfit = float(la.norm(ref - rec, "fro"))
    centered = ref - np.mean(ref, axis=1, keepdims=True)
    denom = float(la.norm(centered, "fro"))
    fit = 0.0 if denom == 0 else float(100 * (1 - (misfit / denom)))
    return fit, misfit


def prepare_dataset_split(
    name_or_path: str,
    test_ratio: float = 0.5,
    standardize: bool = False,
    data_dir: str | Path | None = None,
    delimiter: str | None = None,
    skiprows: int = 0,
    usecols: tuple[int, ...] | None = None,
) -> DatasetSplit:
    path = resolve_data_path(name_or_path, data_dir=data_dir)

    if usecols is None and delimiter is None and skiprows == 0 and path == resolve_data_path(name_or_path, data_dir=data_dir):
        try:
            loaded = load_named_dataset(name_or_path, data_dir=data_dir)
            data = loaded.data
        except FileNotFoundError:
            data = load_matrix(path, delimiter=delimiter, skiprows=skiprows, usecols=usecols)
    else:
        data = load_matrix(path, delimiter=delimiter, skiprows=skiprows, usecols=usecols)

    params: StandardizationParams | None = None
    matrix = np.asarray(data, dtype=float)
    if standardize:
        standardized = standardize_matrix(matrix, axis=0)
        matrix = standardized.data
        params = standardized.params

    split = split_train_test(matrix, test_ratio=test_ratio, axis=0)
    return DatasetSplit(
        source=name_or_path,
        path=_repo_relative_path(path),
        train=np.asarray(split.train, dtype=float),
        test=np.asarray(split.test, dtype=float),
        standardized=standardize,
        standardization_params=params,
    )


def fit_input_output_model(
    train_data: np.ndarray,
    lag: int,
    n_inputs: int,
    x0: int | float | np.ndarray = 1,
    tol: float = 1e-3,
    delta: float = 1e-3,
) -> InputOutputModel:
    w_train = _ensure_2d_rows(train_data)
    result = sparse_lra_sysid(w_train, lag=lag, n_inputs=n_inputs, x0=x0, tol=tol, delta=delta)
    return InputOutputModel(
        lag=lag,
        n_inputs=n_inputs,
        tol=tol,
        delta=delta,
        result=result,
    )


def select_input_output_lag(
    train_data: np.ndarray,
    n_inputs: int,
    lag_candidates: list[int] | np.ndarray,
    x0: int | float | np.ndarray = 1,
    tol: float = 1e-3,
    delta: float = 1e-3,
) -> LagSelectionResult:
    w_train = _ensure_2d_rows(train_data)
    lag_values = np.asarray(sorted(set(int(lag) for lag in lag_candidates if int(lag) >= 2)), dtype=int)
    if lag_values.size == 0:
        raise ValueError("lag_candidates must contain at least one lag >= 2.")

    metric = np.zeros(lag_values.size, dtype=float)
    for i, lag in enumerate(lag_values):
        metric[i] = sparse_lra_sysid(w_train, lag=lag, n_inputs=n_inputs, x0=x0, tol=tol, delta=delta).misfit

    best_idx = int(np.argmin(metric))
    return LagSelectionResult(
        best_lag=int(lag_values[best_idx]),
        best_metric=float(metric[best_idx]),
        lag_values=lag_values,
        metric=metric,
    )


def evaluate_input_output_model(
    model: InputOutputModel,
    full_data: np.ndarray,
    test_horizon: int,
) -> ValidationMetrics:
    w_full = _ensure_2d_rows(full_data)
    if test_horizon < 1 or test_horizon >= w_full.shape[1]:
        raise ValueError("test_horizon must be between 1 and number of time samples - 1.")

    n_outputs = w_full.shape[0] - model.n_inputs
    if n_outputs < 1:
        raise ValueError("Model n_inputs is incompatible with full_data.")

    _, y = cnt.forced_response(
        model.result.sys_ss,
        U=w_full[: model.n_inputs, :],
        X0=model.result.x0,
    )
    y = _ensure_2d_rows(y)

    reference = w_full[-n_outputs:, -test_horizon:]
    predicted = y[:, -test_horizon:]
    fit, misfit = _fit_percent(reference, predicted)
    return ValidationMetrics(
        fit_percent=fit,
        misfit=misfit,
        predicted_outputs=predicted,
        reference_outputs=reference,
    )


def fit_autoregressive_model(
    train_outputs: np.ndarray,
    lag: int,
    tol: float = 1e-3,
    delta: float = 1e-3,
) -> AutoregressiveModel:
    y_train = _ensure_2d_rows(train_outputs)
    result = sparse_lra_aut_sysid(y_train, lag=lag, tol=tol, delta=delta)
    return AutoregressiveModel(
        lag=lag,
        tol=tol,
        delta=delta,
        result=result,
    )


def forecast_autoregressive_model(
    model: AutoregressiveModel,
    history: np.ndarray,
    horizon: int,
) -> AutoregressiveForecast:
    y_history = _ensure_2d_rows(history)
    n_outputs = y_history.shape[0]
    if y_history.shape[1] < model.lag:
        raise ValueError("history must contain at least `lag` time samples.")
    if horizon < 1:
        raise ValueError("horizon must be positive.")

    state = np.asarray(y_history[:, -model.lag :], dtype=float).reshape(n_outputs * model.lag, 1, order="F")
    a = model.result.a
    c = model.result.c
    y_all, _ = _simulate_autonomous(a, c, state, horizon)
    forecast = y_all[:, 1:]
    return AutoregressiveForecast(
        history=y_history,
        horizon=horizon,
        forecast=forecast,
    )


def _simulate_autonomous(a: np.ndarray, c: np.ndarray, x0: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x0, dtype=float).copy()
    y = c.dot(x)
    states = x.copy()

    for _ in range(n_steps):
        x_next = a.dot(x)
        y_next = c.dot(x_next)
        states = np.append(states, x_next, axis=1)
        y = np.append(y, y_next, axis=1)
        x = x_next

    return y, states
