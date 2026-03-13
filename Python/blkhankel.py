#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 00:56:48 2022

@author: kerin
"""
import numpy as np

def blkhank(w: np.ndarray, i: int, j: int) -> np.ndarray:
    if i < 1 or j < 1:
        raise ValueError("i and j must be positive integers.")

    w = np.asarray(w)
    if w.ndim not in (2, 3):
        raise ValueError("w must be a 2D or 3D array.")

    if w.ndim == 3:
        q, n, t = w.shape
        if i + j - 1 > t:
            raise ValueError("i + j - 1 must be <= number of time slices in w.")

        h = np.zeros((i * q, j * n))
        for ii in range(1, i + 1):
            for jj in range(1, j + 1):
                h[(ii - 1) * q : ii * q, (jj - 1) * n : (jj * n)] = w[:, :, ii + jj - 2]
        return h

    q, t = w.shape
    if t < q:
        w = w.T
        q, t = w.shape

    if i + j - 1 > t:
        raise ValueError("i + j - 1 must be <= number of columns in w.")

    h = np.zeros((i * q, j))
    for ii in range(1, i + 1):
        h[((ii - 1) * q) : (ii * q), :] = w[:, (ii - 1) : (ii + j - 1)]
    return h        
    
    
