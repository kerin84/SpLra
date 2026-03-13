#!/usr/bin/env python3
"""
Created on Wed Mar 31 02:57:52 2021
LSSPSOLVER  Sparse linear least squares solver
   Code by Fredy Vides
   For Paper, "Computing Sparse Semilinear Models for Approximately Eventually Periodic Signals"
   by F. Vides
@author: Fredy Vides
"""
import numpy as np
from numpy.linalg import svd,lstsq,norm


def lsspsolver(
    A: np.ndarray,
    Y: np.ndarray,
    tol: float = 1e-3,
    delta: float = 1e-3,
    L: int = 100,
) -> np.ndarray:
    A = np.asarray(A)
    Y = np.asarray(Y)

    if A.ndim != 2 or Y.ndim != 2:
        raise ValueError("A and Y must be 2D arrays.")
    if A.shape[0] != Y.shape[0]:
        raise ValueError("A and Y must have the same number of rows.")
    if tol < 0 or delta < 0:
        raise ValueError("tol and delta must be non-negative.")
    if L < 1:
        raise ValueError("L must be at least 1.")

    n_targets = Y.shape[1]
    n_features = A.shape[1]
    X = np.zeros((n_features, n_targets))

    u, s, v = svd(A, full_matrices=0)
    rk = int(np.sum(s > tol))
    if rk == 0:
        raise ValueError("Effective rank is zero; increase tol or provide better conditioned A.")

    u = u[:, :rk]
    s = np.diag(1 / s[:rk])
    v = v[:rk, :]
    A = np.dot(u.T, A)
    Y = np.dot(u.T, Y)
    X0 = np.dot(v.T, np.dot(s, Y))

    for k in range(n_targets):
        w = np.zeros((n_features,))
        iteration = 1
        error = 1 + tol
        c = X0[:, k]
        x0 = c
        ac = abs(c)
        f = np.argsort(-ac)
        n0 = int(max(sum(ac[f] > delta), 1))
        while iteration <= L and error > tol:
            ff = f[:n0]
            X[:, k] = w
            c = lstsq(A[:, ff], Y[:, k], rcond=None)[0]
            X[ff, k] = c
            error = norm(x0 - X[:, k], np.inf)
            x0 = X[:, k]
            ac = abs(x0)
            f = np.argsort(-ac)
            n0 = int(max(sum(ac[f] > delta), 1))
            iteration = iteration + 1
    return X
