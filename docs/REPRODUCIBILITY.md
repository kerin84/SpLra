# Reproducibility Guide

This guide describes the frozen workflow used to reproduce the official paper artifact and validate the repository after changes.

## 1) Environment Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
export MPLCONFIGDIR="$(pwd)/.mplconfig"
```

Reference environment:

- Python `3.11`
- Runtime dependencies locked in [`requirements-paper.txt`](../requirements-paper.txt)
- Validation tooling locked in [`requirements-dev.txt`](../requirements-dev.txt)

Optional extras for some notebook cells are not part of the locked paper artifact. These include `scikit-learn`, `scienceplots`, and the local `SIPPY-master/` copy used for baseline comparisons.

## 2) Verify Core Numerical Code

Run the regression tests before running notebooks:

```bash
ruff check Python tests scripts
pytest -q tests
python scripts/notebook_smoke_test.py --verbose
```

This validates:

- block-Hankel generation (`blkhankel.py`)
- sparse least-squares solver (`lsspsolver.py`)
- LRA core behavior (`Lra_Modeling.py`)
- AR model identification (`Sparse_Sys_Id.py`)
- core/wrapper compatibility (`Sparse_Lra.py` vs `core/sparse_lra_core.py`)
- I/O preprocessing utilities (`io_layer/sparse_lra_io.py`)

## 3) Recommended Execution Order

Run notebooks in this order for progressive complexity:

1. `Notebooks/HairDryer.ipynb`
2. `Notebooks/Dryer2.ipynb`
3. `Notebooks/cstr.ipynb`
4. `Notebooks/Robot_arm.ipynb`
5. `Notebooks/SISO_sysid.ipynb`
6. `Notebooks/MIMO_sysid.ipynb`
7. `Notebooks/Erie_Lake.ipynb`
8. `Notebooks/Econometric_Model.ipynb`

## 4) Data Source Mapping

- Hair dryer: `Data/dryer.dat`
- Industrial dryer (MIMO): `Data/dryer2.dat`
- CSTR: `Data/cstr.dat`
- Robot arm: `Data/robot_arm.dat`
- Erie lake series: `Data/monthly-lake-erie-levels-1921-19.csv`
- Econometric benchmark: `Data/2022-01.csv`

## 5) Programmatic Pipeline (Without Notebook UI)

Use the new modular layers:

- `io_layer`: data load, standardization, splitting.
- `core`: model estimation and metrics.
- `viz`: plotting only.

Example skeleton:

```python
from io_layer.sparse_lra_io import load_named_dataset, standardize_matrix, split_train_test, to_io_blocks
from core.sparse_lra_core import sparse_lra_sysid

loaded = load_named_dataset("dryer")
scaled = standardize_matrix(loaded.data, axis=0)
split = split_train_test(scaled.data, test_ratio=0.5, axis=0)

# Expected model input shape: (variables, time)
w_train = split.train.T
blocks = to_io_blocks(w_train, n_inputs=1)

result = sparse_lra_sysid(w_train, lag=5, n_inputs=1, x0=0, tol=1e-3, delta=1e-3)
print(result.misfit, result.fit)
```

## 6) Notes On Colab Paths

The notebooks now include a standard setup cell that resolves the repository root locally and keeps compatibility with the current folder layout.

When running locally:

- start Jupyter from the repository root or from `Notebooks/`
- execute the `Reproducibility setup` cell before any modeling cell
- keep `Data/`, `Python/`, and `SIPPY-master/` in their tracked locations

## 7) Expected Outputs

You should obtain:

- fitted system models (transfer function/state-space)
- identification and validation error curves
- lag estimation plots
- simulation and forecasting trajectories with uncertainty bands

Numerical values can vary slightly due to random seeds in forecast sampling.
