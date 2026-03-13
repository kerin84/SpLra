from __future__ import annotations

import numpy as np

from Sparse_Sys_Id import Ar_Model


def test_ar_model_solver_1_recovers_single_lag_coefficient() -> None:
    a_true = 0.8
    n_samples = 140

    y = np.zeros((1, n_samples))
    y[0, 0] = 1.0
    for k in range(1, n_samples):
        y[0, k] = a_true * y[0, k - 1]

    a, hlx, xl, xp, error, rmse_train = Ar_Model(y, L=2, solver=1)

    assert a.shape == (1, 1)
    assert hlx.shape == (1, n_samples - 1)
    assert xl.shape == (1, n_samples - 1)
    assert xp.shape == (1, n_samples - 1)
    assert error.shape == (1, n_samples - 1)
    assert rmse_train.shape == (1,)
    assert np.allclose(a[0, 0], a_true, atol=1e-12)
    assert np.max(np.abs(error)) < 1e-10

def test_ar_model_rejects_unknown_solver() -> None:
    x = np.ones((1, 10))
    try:
        Ar_Model(x, L=2, solver=99)
    except ValueError as exc:
        assert "solver" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported solver")
