from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import control as cnt
import numpy as np
import scipy.linalg as la

from blkhankel import blkhank
from Lra_Modeling import AutR_2_ss, Aut_sys_sim, R2ss, R2tf, lra
from Sparse_Sys_Id import x0_ss_estimate


@dataclass(frozen=True)
class SparseLraSysIdResult:
    misfit: float
    fit: float
    reconstructed: np.ndarray
    residual_operator: np.ndarray
    sys_tf: Any
    sys_ss: Any
    x0: np.ndarray


@dataclass(frozen=True)
class SparseLraAutSysIdResult:
    misfit: float
    fit: float
    a: np.ndarray
    c: np.ndarray
    reconstructed: np.ndarray
    hankel: np.ndarray
    dh: np.ndarray
    x_sparse: np.ndarray


@dataclass(frozen=True)
class ArModelLraResult:
    x_dense: np.ndarray
    x_sparse: np.ndarray
    residual_operator: np.ndarray
    hankel: np.ndarray
    dh: np.ndarray
    predicted: np.ndarray
    error: np.ndarray


@dataclass(frozen=True)
class LagCurveResult:
    lag_values: np.ndarray
    metric: np.ndarray


@dataclass(frozen=True)
class LagArCurveResult:
    lag_values: np.ndarray
    rmse_id: np.ndarray
    rmse_val: np.ndarray


@dataclass(frozen=True)
class ForecastResult:
    t: np.ndarray
    y_forecast: np.ndarray
    y_max: np.ndarray
    y_min: np.ndarray
    y_mean: np.ndarray
    y_total: np.ndarray
    t_total: np.ndarray


def _ensure_2d_rows(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _misfit_and_fit(reference: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float]:
    misfit = float(la.norm(reference - reconstructed, "fro"))
    centered = reference - np.mean(reference, axis=1, keepdims=True)
    denom = float(la.norm(centered, "fro"))
    if denom == 0:
        return misfit, 0.0
    fit = 100 * (1 - (misfit / denom))
    return misfit, float(fit)


def sparse_lra_sysid(
    w: np.ndarray,
    lag: int,
    n_inputs: int,
    x0: int | float | np.ndarray = 1,
    tol: float = 1e-3,
    delta: float = 1e-3,
) -> SparseLraSysIdResult:
    w = np.asarray(w)
    if w.ndim != 2:
        raise ValueError("w must be a 2D array.")

    n_rows, n_cols = w.shape
    n_outputs = n_rows - n_inputs
    if n_outputs < 1:
        raise ValueError("n_inputs must be smaller than the number of rows in w.")

    u, y = w[:n_inputs, :], w[n_inputs:, :]
    hr = blkhank(w, lag + 1, n_cols - lag)
    _, r_sp, _, dh, _, x_sp = lra(hr, hr.shape[0] - n_outputs, tol, delta)

    sys_tf = R2tf(r_sp, lag, n_inputs, dt=1)
    sys_ss = R2ss(r_sp, n_inputs, lag, 1)

    if isinstance(x0, np.ndarray):
        x0_est = np.asarray(x0)
    elif x0 == 0:
        x0_est = np.zeros((lag * n_outputs, 1))
    else:
        x0_est = x0_ss_estimate(sys_ss, lag + 1, y, u)

    uh = np.block([u[:, :lag], dh[-n_rows:-n_outputs, :]])
    _, yr = cnt.forced_response(sys_ss, U=uh, X0=x0_est)
    yr = _ensure_2d_rows(yr)

    reconstructed = np.block([[uh], [yr]])
    misfit, fit = _misfit_and_fit(w, reconstructed)

    return SparseLraSysIdResult(
        misfit=misfit,
        fit=fit,
        reconstructed=reconstructed,
        residual_operator=r_sp,
        sys_tf=sys_tf,
        sys_ss=sys_ss,
        x0=x0_est,
    )


def sparse_lra_aut_sysid(
    w: np.ndarray,
    lag: int,
    tol: float = 1e-3,
    delta: float = 1e-3,
) -> SparseLraAutSysIdResult:
    w = np.asarray(w)
    if w.ndim != 2:
        raise ValueError("w must be a 2D array.")

    n_rows, n_cols = w.shape
    hr = blkhank(w, lag + 1, n_cols - lag)
    _, _, _, dh, _, x_sp = lra(hr, hr.shape[0] - n_rows, tol, delta)

    a, c = AutR_2_ss(x_sp, 1)
    reconstructed, _ = Aut_sys_sim(a, c, hr[:-n_rows, 0:1], n_cols - lag - 1)
    reconstructed = np.block([w[:, :lag], reconstructed])

    misfit, fit = _misfit_and_fit(w, reconstructed)

    return SparseLraAutSysIdResult(
        misfit=misfit,
        fit=fit,
        a=a,
        c=c,
        reconstructed=reconstructed,
        hankel=hr,
        dh=dh,
        x_sparse=x_sp,
    )


def ar_model_lra(
    x: np.ndarray,
    lag: int,
    tol: float,
    delta: float,
) -> ArModelLraResult:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")

    n_rows, n_cols = x.shape
    h = blkhank(x, lag + 1, n_cols - lag)

    x_dense, r_sp, _, dh, _, x_sp = lra(h, h.shape[0] - n_rows, tol, delta)
    hlx = dh[: n_rows * lag, :]

    predicted = x_sp.dot(hlx)
    error = r_sp.dot(h)

    return ArModelLraResult(
        x_dense=x_dense,
        x_sparse=x_sp,
        residual_operator=r_sp,
        hankel=h,
        dh=dh,
        predicted=predicted,
        error=error,
    )


def compute_lag_misfit(
    w: np.ndarray,
    lag_max: int,
    n_inputs: int,
    x0: int | float | np.ndarray = 1,
    tol: float = 1e-3,
    delta: float = 1e-3,
) -> LagCurveResult:
    lag_values = np.arange(2, lag_max)
    misfit = np.zeros(len(lag_values))

    for lag in lag_values:
        misfit[lag - 2] = sparse_lra_sysid(w, lag, n_inputs, x0, tol, delta).misfit

    return LagCurveResult(lag_values=lag_values, metric=misfit)


def compute_lag_misfit_aut(
    w: np.ndarray,
    lag_max: int,
    tol: float,
    delta: float,
) -> LagCurveResult:
    lag_values = np.arange(2, lag_max)
    misfit = np.zeros(len(lag_values))

    for lag in lag_values:
        misfit[lag - 2] = sparse_lra_aut_sysid(w, lag, tol, delta).misfit

    return LagCurveResult(lag_values=lag_values, metric=misfit)


def compute_lag_rmse(
    yid: np.ndarray,
    yval: np.ndarray,
    lag_max: int,
    tol: float,
    delta: float,
) -> LagArCurveResult:
    lag_values = np.arange(2, lag_max)
    e_id = np.zeros(len(lag_values))
    e_val = np.zeros(len(lag_values))

    for lag in lag_values:
        model = ar_model_lra(yid, lag, tol, delta)
        e_id[lag - 2] = la.norm(model.error) / np.sqrt(yid.shape[1] - lag)

        h_val = blkhank(yval, lag + 1, yval.shape[1] - lag)
        e_val[lag - 2] = la.norm(model.residual_operator.dot(h_val)) / np.sqrt(yval.shape[1] - lag)

    return LagArCurveResult(lag_values=lag_values, rmse_id=e_id, rmse_val=e_val)


def ar_lra_forecast(
    sys_ss: Any,
    yid: np.ndarray,
    e_id: np.ndarray,
    x0: np.ndarray,
    sigma2: float,
    n_paths: int,
    n_horizon: int,
    output_index: int,
) -> ForecastResult:
    n_outputs, n_samples = e_id.shape
    t, y, x = cnt.forced_response(sys_ss, U=e_id, X0=x0, return_x=True)
    y = _ensure_2d_rows(y)

    x = np.asarray(x)
    if x.ndim == 1:
        xfcast = x[-1:]
    else:
        xfcast = x[:, -1]

    yj = y[output_index : output_index + 1, :]
    y_forecast = np.zeros((yid.shape[0], n_horizon, n_paths))

    for i in range(n_paths):
        e_val = sigma2 * np.random.randn(n_outputs, n_horizon)
        _, y_path = cnt.forced_response(sys_ss, U=e_val, X0=xfcast)
        y_path = _ensure_2d_rows(y_path)
        y_forecast[:, :, i] = y_path

    y_max = np.max(y_forecast[output_index : output_index + 1, :, :], axis=2)
    y_min = np.min(y_forecast[output_index : output_index + 1, :, :], axis=2)
    y_mean = np.mean(y_forecast[output_index : output_index + 1, :, :], axis=2)

    y_total = np.append(yj, y_mean)
    t1 = np.arange(t[-1] + 1, t[-1] + n_horizon + 1)
    t_total = np.append(t, t1)

    return ForecastResult(
        t=np.asarray(t),
        y_forecast=y_forecast,
        y_max=y_max,
        y_min=y_min,
        y_mean=y_mean,
        y_total=y_total,
        t_total=t_total,
    )
