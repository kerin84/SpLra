from __future__ import annotations

import numpy as np

from lsspsolver import lsspsolver


def test_lsspsolver_reconstructs_sparse_solution() -> None:
    a = np.array(
        [
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [1.0, 1.0, 0.5, 0.5],
            [2.0, 1.0, 0.0, 0.5],
            [1.0, 2.0, 0.5, 0.0],
        ]
    )
    x_true = np.array(
        [
            [1.5, 0.0],
            [0.0, -2.0],
            [0.0, 1.25],
            [0.5, 0.0],
        ]
    )
    y = a @ x_true

    x_est = lsspsolver(a, y, tol=1e-10, delta=1e-8, L=80)

    assert x_est.shape == x_true.shape
    assert np.allclose(a @ x_est, y, atol=1e-7)
    assert np.allclose(x_est, x_true, atol=1e-6)

def test_lsspsolver_rejects_rank_zero_projection() -> None:
    a = np.zeros((3, 2))
    y = np.zeros((3, 1))
    try:
        lsspsolver(a, y, tol=1e-3, delta=1e-3, L=5)
    except ValueError as exc:
        assert "Effective rank" in str(exc)
    else:
        raise AssertionError("Expected ValueError for rank-zero matrix")
