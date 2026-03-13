# Official Code Repository
## Sparse Semilinear Dynamic Models in Python for Identifying and Forecasting Dynamic Systems

This repository contains the official implementation used in the article:

- K. Cardona, F. Vides (2024).
- *Sparse Semilinear Dynamic Models in Python for Identifying and Forecasting Dynamic Systems*.
- Revista de la Escuela de Fisica (REF-UNAH), Vol. 12, No. 1, pp. 66-81.
- DOI: `10.5377/ref.v12i1.19433`

Article page: <https://fisica.unah.edu.hn/publicaciones/revista-ref/ultimo-volumen-ref/ref-unah-12-1-4/>

## What Is In This Repository

- `Python/`: core modeling code (sparse system identification, LRA modeling, utilities).
- `Notebooks/`: experiment notebooks used for case studies and comparisons.
- `Data/`: datasets used by notebooks and experiments.
- `SIPPY-master/`: local copy of SIPPY used for baseline/comparison workflows.
- `tests/`: unit tests for critical numerical components.

## Quick Start

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy control statsmodels scikit-learn matplotlib pytest ruff
# Optional (plot style used in notebooks)
pip install scienceplots
```

### 2) Run tests

```bash
pytest -q tests
```

### 3) Open notebooks

Run notebooks in `Notebooks/` using Jupyter or Colab (most notebooks were originally authored for Colab paths).

## Main Python Modules

- `Python/Sparse_Sys_Id.py`: AR/AR-IO identification, simulation, and state-space conversion utilities.
- `Python/Lra_Modeling.py`: low-rank approximation routines and model conversion to transfer function/state-space forms.
- `Python/lsspsolver.py`: sparse linear least-squares solver.
- `Python/blkhankel.py`: block-Hankel matrix construction.
- `Python/Sparse_Lra.py`: legacy compatibility API for sparse-LRA workflows.
- `Python/core/sparse_lra_core.py`: pure computation layer with typed result objects.
- `Python/viz/sparse_lra_plots.py`: plotting functions separated from core logic.
- `Python/io_layer/sparse_lra_io.py`: dataset loading, standardization, and train/test split utilities.

## Datasets

Available local datasets:

- `Data/dryer.dat`
- `Data/dryer2.dat`
- `Data/cstr.dat`
- `Data/robot_arm.dat`
- `Data/monthly-lake-erie-levels-1921-19.csv`
- `Data/2022-01.csv`

## Reproducibility

See [Reproducibility Guide](docs/REPRODUCIBILITY.md) for the recommended sequence to reproduce experiments and forecasting workflows.

## API And Architecture Notes

See [API Reference](docs/API_REFERENCE.md) for the current module organization (`core`, `io_layer`, `viz`) and backward compatibility details.

## Citation

If you use this repository, please cite the article above and optionally include the metadata in [`CITATION.cff`](CITATION.cff).
