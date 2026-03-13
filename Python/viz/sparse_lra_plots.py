from __future__ import annotations

from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np

from core.sparse_lra_core import ForecastResult

try:
    import scienceplots  # noqa: F401
    _HAS_SCIENCEPLOTS = True
except ModuleNotFoundError:
    _HAS_SCIENCEPLOTS = False


def _style_context():
    if _HAS_SCIENCEPLOTS:
        return plt.style.context(["science"])
    return nullcontext()


def plot_lag_misfit(lag_values: np.ndarray, misfit: np.ndarray) -> None:
    with _style_context():
        plt.figure(figsize=(9, 3))
        plt.plot(lag_values, misfit, ".-", label="$misfit$")
        plt.title("Estimacion del Lag")
        plt.xlabel("$L$")
        plt.ylabel("$misfit$")
        plt.legend()
        plt.show()


def plot_lag_misfit_aut(lag_values: np.ndarray, misfit: np.ndarray) -> None:
    plt.figure(figsize=(6, 2.5))
    plt.plot(lag_values, misfit, ".-")
    plt.show()


def plot_lag_rmse(lag_values: np.ndarray, rmse_id: np.ndarray, rmse_val: np.ndarray) -> None:
    with _style_context():
        plt.figure(figsize=(9, 3))
        plt.plot(lag_values, rmse_id, ".-", label="$\\epsilon_{id}$")
        plt.plot(lag_values, rmse_val, ".-", color="r", label="$\\epsilon_{val}$")
        plt.title("Estimacion del Lag")
        plt.xlabel("$L$")
        plt.ylabel("rmse($\\epsilon$)")
        plt.legend()
        plt.show()


def plot_forecast_band(
    result: ForecastResult,
    yval: np.ndarray,
    output_index: int,
    n_horizon: int,
) -> None:
    yval = np.asarray(yval)
    with _style_context():
        plt.figure(figsize=(9, 3.5))
        plt.plot(result.t_total[-2 * n_horizon :], result.y_total[-2 * n_horizon :], label="$y_{sim}$")
        plt.plot(
            np.arange(result.t[-1] + 1, result.t[-1] + n_horizon + 1),
            yval[output_index, :],
            "--",
            color="r",
            label="$y_{val}$",
        )
        plt.fill_between(
            np.arange(result.t[-1] + 1, result.t[-1] + n_horizon + 1),
            result.y_min[0, :],
            result.y_max[0, :],
            facecolor="k",
            alpha=0.17,
        )
        plt.title("Simulacion + Prediccion")
        plt.xlabel("$t$")
        plt.ylabel("Amplitud")
        plt.legend()
        plt.show()
