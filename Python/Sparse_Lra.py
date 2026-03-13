# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 22:26:10 2023

@author: kerin
"""

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401
    _HAS_SCIENCEPLOTS = True
except ModuleNotFoundError:
    _HAS_SCIENCEPLOTS = False

from core.sparse_lra_core import (
    ar_lra_forecast,
    ar_model_lra,
    compute_lag_misfit,
    compute_lag_misfit_aut,
    compute_lag_rmse,
    sparse_lra_aut_sysid,
    sparse_lra_sysid,
)
from viz.sparse_lra_plots import (
    plot_forecast_band,
    plot_lag_misfit,
    plot_lag_misfit_aut,
    plot_lag_rmse,
)

warnings.filterwarnings("ignore")

if _HAS_SCIENCEPLOTS:
    plt.style.use("science")
mpl.rcParams["lines.linewidth"] = 0.6


def Sparse_lra_sysid(w, L, m, x0=1, tol=1e-3, delta=1e-3):
    result = sparse_lra_sysid(w, lag=L, n_inputs=m, x0=x0, tol=tol, delta=delta)
    return (
        result.misfit,
        result.fit,
        result.reconstructed,
        result.residual_operator,
        result.sys_tf,
        result.sys_ss,
        result.x0,
    )


def Sparse_lra_Aut_sysid(w, L, tol=1e-3, delta=1e-3):
    result = sparse_lra_aut_sysid(w, lag=L, tol=tol, delta=delta)
    return (
        result.misfit,
        result.fit,
        result.a,
        result.c,
        result.reconstructed,
        result.hankel,
        result.dh,
        result.x_sparse,
    )


def Ar_Model_lra(X, L, tol, delta):
    result = ar_model_lra(X, lag=L, tol=tol, delta=delta)
    return (
        result.x_dense,
        result.x_sparse,
        result.residual_operator,
        result.hankel,
        result.dh,
        result.predicted,
        result.error,
    )


def tol_delta_tuning(w, L, m):
    x = np.linspace(0.0, 0.3, 300)
    M = np.zeros(len(x))

    for i in np.arange(len(x)):
        M[i] = Sparse_lra_sysid(w, L, m, x[i], x[i])[0]

    plt.plot(x, M)


def lag_est(w, Lmax, m, x0=1, tol=1e-3, delta=1e-3):
    result = compute_lag_misfit(w, lag_max=Lmax, n_inputs=m, x0=x0, tol=tol, delta=delta)
    plot_lag_misfit(result.lag_values, result.metric)


def lag_est_aut(w, Lmax, tol, delta):
    result = compute_lag_misfit_aut(w, lag_max=Lmax, tol=tol, delta=delta)
    plot_lag_misfit_aut(result.lag_values, result.metric)


def lag_est_ar(yid, yval, Lmax, tol, delta):
    result = compute_lag_rmse(yid, yval, lag_max=Lmax, tol=tol, delta=delta)
    plot_lag_rmse(result.lag_values, result.rmse_id, result.rmse_val)


def Ar_lra_forecast(sys_ss, yid, yval, e_id, x0, sigma2, M, N, j):
    forecast = ar_lra_forecast(
        sys_ss=sys_ss,
        yid=yid,
        e_id=e_id,
        x0=x0,
        sigma2=sigma2,
        n_paths=M,
        n_horizon=N,
        output_index=j,
    )
    plot_forecast_band(forecast, yval=yval, output_index=j, n_horizon=N)
    return forecast.t, forecast.y_forecast


def sparsity_ss(sys_ss):
    X = np.block([[sys_ss.A, sys_ss.B], [sys_ss.C, sys_ss.D]])
    spar = (np.count_nonzero(X) / X.size) * 100
    return spar
