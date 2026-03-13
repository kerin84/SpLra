# API And Architecture Reference

## Current Organization

### `Python/core`
Pure numerical logic (no plotting side effects).

Primary module:

- `core/sparse_lra_core.py`

Main typed outputs:

- `SparseLraSysIdResult`
- `SparseLraAutSysIdResult`
- `ArModelLraResult`
- `LagCurveResult`
- `LagArCurveResult`
- `ForecastResult`

Main functions:

- `sparse_lra_sysid(...)`
- `sparse_lra_aut_sysid(...)`
- `ar_model_lra(...)`
- `compute_lag_misfit(...)`
- `compute_lag_misfit_aut(...)`
- `compute_lag_rmse(...)`
- `ar_lra_forecast(...)`

### `Python/io_layer`
Input/output and preprocessing utilities.

Primary module:

- `io_layer/sparse_lra_io.py`

Main functions:

- `resolve_data_path(...)`
- `load_matrix(...)`
- `load_named_dataset(...)`
- `standardize_matrix(...)`
- `apply_standardization(...)`
- `invert_standardization(...)`
- `split_train_test(...)`
- `to_io_blocks(...)`

### `Python/viz`
Plot-only layer.

Primary module:

- `viz/sparse_lra_plots.py`

Main functions:

- `plot_lag_misfit(...)`
- `plot_lag_misfit_aut(...)`
- `plot_lag_rmse(...)`
- `plot_forecast_band(...)`

## Legacy Compatibility

`Python/Sparse_Lra.py` keeps the original public API and returns tuple outputs expected by existing notebooks:

- `Sparse_lra_sysid(...)`
- `Sparse_lra_Aut_sysid(...)`
- `Ar_Model_lra(...)`
- `lag_est(...)`
- `lag_est_aut(...)`
- `lag_est_ar(...)`
- `Ar_lra_forecast(...)`
- `sparsity_ss(...)`

Internally, these wrappers delegate to `core` and `viz`.

## Design Rules

- Core functions should not call plotting functions.
- Plot modules should receive precomputed arrays/results.
- New notebook logic should prefer `io_layer` + `core` + `viz` directly.
- Keep wrapper behavior stable for backward compatibility.

## Testing Policy

Any change in:

- `blkhankel.py`
- `lsspsolver.py`
- `Lra_Modeling.py`
- `Sparse_Sys_Id.py`
- `core/sparse_lra_core.py`
- `io_layer/sparse_lra_io.py`

must be validated with:

```bash
pytest -q tests
ruff check Python tests scripts
python scripts/notebook_smoke_test.py --verbose
```
