from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import control as cnt
import numpy as np
import scipy.linalg as la

from core.sparse_lra_core import sparse_lra_sysid
from io_layer.sparse_lra_io import load_named_dataset, split_train_test, standardize_matrix


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    lag: int
    n_inputs: int
    test_ratio: float
    tol: float
    delta: float
    standardize_outputs: bool = False


@dataclass(frozen=True)
class ExperimentMetrics:
    name: str
    dataset_path: str
    lag: int
    n_inputs: int
    n_train_samples: int
    n_test_samples: int
    identification_misfit: float
    identification_fit_percent: float
    train_output_fit_percent: float
    validation_fit_percent: float
    sparsity_percent: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _ensure_2d_rows(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _fit_percent(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    rec = np.asarray(reconstructed, dtype=float)
    misfit = float(la.norm(ref - rec, "fro"))
    centered = ref - np.mean(ref, axis=1, keepdims=True)
    denom = float(la.norm(centered, "fro"))
    if denom == 0:
        return 0.0
    return float(100 * (1 - (misfit / denom)))


def _forced_outputs(sys_ss: object, u: np.ndarray, x0: np.ndarray) -> np.ndarray:
    _, y = cnt.forced_response(sys_ss, U=u, X0=x0)
    return _ensure_2d_rows(y)


def _state_space_sparsity_percent(sys_ss: object) -> float:
    x = np.block([[sys_ss.A, sys_ss.B], [sys_ss.C, sys_ss.D]])
    return float((np.count_nonzero(x) / x.size) * 100)


def _repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_project_root()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _run_sysid_experiment(
    name: str,
    data: np.ndarray,
    dataset_path: Path,
    config: ExperimentConfig,
) -> ExperimentMetrics:
    split = split_train_test(data, test_ratio=config.test_ratio, axis=0)
    w_train = split.train.T
    w_test = split.test.T
    w_full = data.T

    result = sparse_lra_sysid(
        w_train,
        lag=config.lag,
        n_inputs=config.n_inputs,
        tol=config.tol,
        delta=config.delta,
    )

    y_full = _forced_outputs(result.sys_ss, u=w_full[: config.n_inputs, :], x0=result.x0)
    y_test = y_full[:, -w_test.shape[1] :]

    n_outputs = w_full.shape[0] - config.n_inputs
    train_ref = w_train[-n_outputs:, :]
    train_rec = result.reconstructed[-n_outputs:, :]
    test_ref = w_test[-n_outputs:, :]

    return ExperimentMetrics(
        name=name,
        dataset_path=_repo_relative_path(dataset_path),
        lag=config.lag,
        n_inputs=config.n_inputs,
        n_train_samples=int(w_train.shape[1]),
        n_test_samples=int(w_test.shape[1]),
        identification_misfit=float(result.misfit),
        identification_fit_percent=float(result.fit),
        train_output_fit_percent=_fit_percent(train_ref, train_rec),
        validation_fit_percent=_fit_percent(test_ref, y_test),
        sparsity_percent=_state_space_sparsity_percent(result.sys_ss),
    )


def run_hair_dryer_experiment() -> ExperimentMetrics:
    config = ExperimentConfig(
        dataset="dryer",
        lag=5,
        n_inputs=1,
        test_ratio=0.5,
        tol=1e-2,
        delta=1e-2,
    )
    loaded = load_named_dataset(config.dataset)
    return _run_sysid_experiment(
        name="hair_dryer",
        data=loaded.data,
        dataset_path=loaded.path,
        config=config,
    )


def run_cstr_experiment() -> ExperimentMetrics:
    config = ExperimentConfig(
        dataset="cstr",
        lag=1,
        n_inputs=1,
        test_ratio=0.5,
        tol=1e-2,
        delta=1e-2,
        standardize_outputs=True,
    )
    loaded = load_named_dataset(config.dataset)
    standardized = standardize_matrix(loaded.data[:, 1:], axis=0)
    return _run_sysid_experiment(
        name="cstr",
        data=standardized.data,
        dataset_path=loaded.path,
        config=config,
    )


def run_paper_experiments() -> dict[str, dict[str, float | int | str]]:
    experiments = [
        run_hair_dryer_experiment(),
        run_cstr_experiment(),
    ]
    return {experiment.name: experiment.to_dict() for experiment in experiments}
