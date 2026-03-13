from __future__ import annotations

import numpy as np

from Sparse_Lra import Ar_Model_lra, Sparse_lra_sysid
from core.sparse_lra_core import ar_model_lra, compute_lag_misfit, sparse_lra_sysid


def _build_io_dataset(n_samples: int = 120) -> np.ndarray:
    rng = np.random.default_rng(7)
    u = rng.normal(scale=0.6, size=n_samples)
    y = np.zeros(n_samples)
    for k in range(1, n_samples):
        y[k] = 0.72 * y[k - 1] + 0.2 * u[k - 1]
    return np.vstack([u, y])


def test_sparse_lra_sysid_wrapper_matches_core_output() -> None:
    w = _build_io_dataset()

    core_result = sparse_lra_sysid(w, lag=5, n_inputs=1, x0=0, tol=1e-3, delta=1e-3)
    legacy_result = Sparse_lra_sysid(w, L=5, m=1, x0=0, tol=1e-3, delta=1e-3)

    assert np.isclose(legacy_result[0], core_result.misfit)
    assert np.isclose(legacy_result[1], core_result.fit)
    assert np.allclose(legacy_result[2], core_result.reconstructed)
    assert np.allclose(legacy_result[3], core_result.residual_operator)
    assert np.allclose(legacy_result[6], core_result.x0)


def test_ar_model_lra_wrapper_matches_core_output() -> None:
    y = _build_io_dataset()[1:2, :]

    core_result = ar_model_lra(y, lag=4, tol=1e-3, delta=1e-3)
    legacy_result = Ar_Model_lra(y, L=4, tol=1e-3, delta=1e-3)

    assert np.allclose(legacy_result[0], core_result.x_dense)
    assert np.allclose(legacy_result[1], core_result.x_sparse)
    assert np.allclose(legacy_result[2], core_result.residual_operator)
    assert np.allclose(legacy_result[5], core_result.predicted)
    assert np.allclose(legacy_result[6], core_result.error)


def test_compute_lag_misfit_returns_expected_grid() -> None:
    w = _build_io_dataset()

    result = compute_lag_misfit(w, lag_max=9, n_inputs=1, x0=0, tol=1e-3, delta=1e-3)

    assert np.array_equal(result.lag_values, np.arange(2, 9))
    assert result.metric.shape == (7,)
    assert np.all(np.isfinite(result.metric))
