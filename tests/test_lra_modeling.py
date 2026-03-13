from __future__ import annotations

import numpy as np

from Lra_Modeling import lra


def test_lra_returns_consistent_shapes_and_sparse_residual_operator() -> None:
    rng = np.random.default_rng(42)
    d = rng.normal(size=(6, 10))
    r = 4

    r_dense, r_sparse, p, dh, x_dense, x_sparse = lra(d, r, tol=1e-8, delta=1e-8)

    assert r_dense.shape == (2, 6)
    assert r_sparse.shape == (2, 6)
    assert p.shape == (6, 4)
    assert dh.shape == (6, 10)
    assert x_dense.shape == (2, 4)
    assert x_sparse.shape == (2, 4)

    # El operador sparse debe reducir significativamente la energía en la aproximación de rango-r.
    reduced = np.linalg.norm(r_sparse @ dh)
    baseline = np.linalg.norm(dh)
    assert reduced < baseline * 0.2
