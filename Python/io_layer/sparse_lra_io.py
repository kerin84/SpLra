from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    path: Path
    data: np.ndarray


@dataclass(frozen=True)
class StandardizationParams:
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class StandardizedMatrix:
    data: np.ndarray
    params: StandardizationParams


@dataclass(frozen=True)
class TrainTestSplit:
    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class IoBlocks:
    inputs: np.ndarray
    outputs: np.ndarray


_DEFAULT_DATASET_FILES: dict[str, str] = {
    "dryer": "dryer.dat",
    "dryer2": "dryer2.dat",
    "cstr": "cstr.dat",
    "robot_arm": "robot_arm.dat",
    "erie_lake": "monthly-lake-erie-levels-1921-19.csv",
}

_DATASET_LOAD_OPTIONS: dict[str, dict[str, object]] = {
    "erie_lake": {"delimiter": ",", "skiprows": 1, "usecols": (1,)},
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return _project_root() / "Data"


def resolve_data_path(name_or_path: str, data_dir: str | Path | None = None) -> Path:
    raw = Path(name_or_path)
    if raw.is_absolute() or raw.suffix:
        candidate = raw
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Dataset path not found: {candidate}")

    base_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    base_dir = base_dir.resolve()

    mapped_name = _DEFAULT_DATASET_FILES.get(name_or_path, name_or_path)
    candidate_names = [mapped_name]
    if "." not in mapped_name:
        candidate_names.extend([f"{mapped_name}.dat", f"{mapped_name}.csv"])

    for candidate_name in candidate_names:
        candidate = base_dir / candidate_name
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"No dataset found for '{name_or_path}' in {base_dir}")


def load_matrix(
    path: str | Path,
    delimiter: str | None = None,
    skiprows: int = 0,
    usecols: tuple[int, ...] | None = None,
    ndmin: int = 2,
) -> np.ndarray:
    array = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows, usecols=usecols, ndmin=ndmin)
    return np.asarray(array, dtype=float)


def load_named_dataset(name: str, data_dir: str | Path | None = None) -> LoadedDataset:
    path = resolve_data_path(name, data_dir=data_dir)

    options = _DATASET_LOAD_OPTIONS.get(name, {})
    data = load_matrix(path=path, **options)
    return LoadedDataset(name=name, path=path, data=data)


def standardize_matrix(data: np.ndarray, axis: int = 0, eps: float = 1e-12) -> StandardizedMatrix:
    arr = np.asarray(data, dtype=float)
    if arr.ndim < 1:
        raise ValueError("data must have at least 1 dimension.")

    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    std_safe = np.where(std < eps, 1.0, std)

    standardized = (arr - mean) / std_safe
    return StandardizedMatrix(data=standardized, params=StandardizationParams(mean=mean, std=std_safe))


def apply_standardization(data: np.ndarray, params: StandardizationParams) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    return (arr - params.mean) / params.std


def invert_standardization(data: np.ndarray, params: StandardizationParams) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    return arr * params.std + params.mean


def split_train_test(data: np.ndarray, test_ratio: float = 0.5, axis: int = 0) -> TrainTestSplit:
    if not (0 < test_ratio < 1):
        raise ValueError("test_ratio must be between 0 and 1.")

    arr = np.asarray(data)
    axis = arr.ndim + axis if axis < 0 else axis
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("axis is out of bounds for data dimensions.")

    n = arr.shape[axis]
    n_test = int(test_ratio * n)
    n_test = max(1, min(n - 1, n_test))
    split_idx = n - n_test

    slicer_train = [slice(None)] * arr.ndim
    slicer_test = [slice(None)] * arr.ndim
    slicer_train[axis] = slice(0, split_idx)
    slicer_test[axis] = slice(split_idx, None)

    train = arr[tuple(slicer_train)]
    test = arr[tuple(slicer_test)]
    return TrainTestSplit(train=train, test=test)


def to_io_blocks(w: np.ndarray, n_inputs: int) -> IoBlocks:
    arr = np.asarray(w, dtype=float)
    if arr.ndim != 2:
        raise ValueError("w must be a 2D array with shape (variables, time).")
    if not (1 <= n_inputs < arr.shape[0]):
        raise ValueError("n_inputs must satisfy 1 <= n_inputs < number of rows in w.")

    return IoBlocks(inputs=arr[:n_inputs, :], outputs=arr[n_inputs:, :])
