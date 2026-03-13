from __future__ import annotations

import numpy as np

from io_layer.sparse_lra_io import (
    apply_standardization,
    invert_standardization,
    load_named_dataset,
    split_train_test,
    standardize_matrix,
    to_io_blocks,
)


def test_load_named_dataset_dryer_has_expected_shape() -> None:
    dataset = load_named_dataset("dryer")
    assert dataset.data.shape == (1000, 2)


def test_standardize_and_inverse_roundtrip() -> None:
    x = np.array([[1.0, 2.0], [3.0, 5.0], [5.0, 8.0]])
    standardized = standardize_matrix(x, axis=0)

    reconstructed = invert_standardization(standardized.data, standardized.params)
    reapplied = apply_standardization(x, standardized.params)

    assert np.allclose(reconstructed, x)
    assert np.allclose(reapplied, standardized.data)


def test_split_train_test_axis_control() -> None:
    x = np.arange(20).reshape(10, 2)

    split0 = split_train_test(x, test_ratio=0.3, axis=0)
    assert split0.train.shape == (7, 2)
    assert split0.test.shape == (3, 2)

    split1 = split_train_test(x.T, test_ratio=0.25, axis=1)
    assert split1.train.shape == (2, 8)
    assert split1.test.shape == (2, 2)


def test_to_io_blocks_splits_rows() -> None:
    w = np.arange(24, dtype=float).reshape(4, 6)
    blocks = to_io_blocks(w, n_inputs=2)

    assert blocks.inputs.shape == (2, 6)
    assert blocks.outputs.shape == (2, 6)
    assert np.allclose(blocks.inputs, w[:2, :])
    assert np.allclose(blocks.outputs, w[2:, :])
