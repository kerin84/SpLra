# Reproducibility Guide

This guide describes a practical, stable workflow to reproduce the experiments associated with the paper.

## 1) Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy control statsmodels scikit-learn matplotlib pytest ruff jupyter
# Optional visual style
pip install scienceplots
```

## 2) Verify Core Numerical Code

Run the regression tests before running notebooks:

```bash
pytest -q tests
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

Several notebooks were authored in Google Colab with absolute Drive paths (`/content/drive/MyDrive/...`).

When running locally:

- replace absolute Colab paths with local relative paths (`Data/...`)
- remove package installation cells if dependencies are already installed

## 7) Expected Outputs

You should obtain:

- fitted system models (transfer function/state-space)
- identification and validation error curves
- lag estimation plots
- simulation and forecasting trajectories with uncertainty bands

Numerical values can vary slightly due to random seeds in forecast sampling.
