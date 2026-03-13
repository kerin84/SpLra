from __future__ import annotations

import numpy as np

from blkhankel import blkhank


def test_blkhank_2d_builds_expected_hankel_blocks() -> None:
    w = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])

    got = blkhank(w, i=2, j=3)
    expected = np.array(
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [2.0, 3.0, 4.0], [20.0, 30.0, 40.0]]
    )

    assert got.shape == (4, 3)
    assert np.allclose(got, expected)


def test_blkhank_2d_transposes_when_time_is_in_rows() -> None:
    w = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])

    got = blkhank(w, i=2, j=3)
    expected = np.array(
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [2.0, 3.0, 4.0], [20.0, 30.0, 40.0]]
    )

    assert got.shape == (4, 3)
    assert np.allclose(got, expected)


def test_blkhank_3d_builds_block_hankel_matrix() -> None:
    w = np.array(
        [
            [
                [1.0, 3.0, 5.0, 7.0],
                [2.0, 4.0, 6.0, 8.0],
            ]
        ]
    )

    got = blkhank(w, i=2, j=2)
    expected = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])

    assert got.shape == (2, 4)
    assert np.allclose(got, expected)

def test_blkhank_rejects_invalid_window_size() -> None:
    w = np.ones((2, 4))
    try:
        blkhank(w, i=0, j=2)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid i")
