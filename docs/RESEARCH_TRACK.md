# Research Track

This document defines how the repository evolves beyond the frozen paper artifact.

## Two Tracks

### Paper Artifact

The paper artifact is the stable reference surface used to reproduce the published REF-UNAH results.

It includes:

- `artifacts/paper_experiment_baselines.json`
- `scripts/run_paper_experiments.py`
- notebooks and datasets required for the published workflows
- the locked environments in `requirements-paper.txt` and `requirements-dev.txt`

Changes in this track should preserve published behavior or improve reproducibility without changing the scientific claim.

### Research Track

The research track is the exploratory surface for new datasets, baselines, tuning strategies, robustness studies, and API growth.

It includes:

- `Python/api/`
- `Python/research/`
- `scripts/run_research_benchmarks.py`

Changes in this track may add new metrics, new experimental protocols, and new abstractions, but they should not silently rewrite the paper artifact.

## Contribution Rules

- Keep paper reproducibility checks passing before merging research changes.
- Add new experiments through `Python/research/` or `scripts/`, not by embedding logic directly into notebooks.
- Treat `Python/Sparse_Lra.py` as a compatibility surface; new code should target `api`, `core`, `io_layer`, `viz`, or `research`.
- When introducing a new benchmark, define:
  - dataset
  - preprocessing
  - train/test split
  - baseline methods
  - primary metric
- When introducing a robustness claim, specify:
  - noise model
  - sample-size protocol
  - hyperparameter sweep
  - random seed

## Initial Research Protocol

The initial benchmark suite uses two paper-aligned cases:

- `hair_dryer`
- `cstr`

For each case, the suite reports:

- sparse-LRA validation fit
- persistence baseline validation fit
- mean baseline validation fit
- improvement over the best simple baseline
- robustness to additive Gaussian noise
- robustness to reduced training sample size
- sensitivity to lag choices near the canonical configuration

Run it with:

```bash
export XDG_CACHE_HOME="$(pwd)/.cache"
export MPLCONFIGDIR="$(pwd)/.mplconfig"
python scripts/run_research_benchmarks.py --write-json artifacts/research_benchmark_results.json
```

The repository currently includes a generated snapshot in `artifacts/research_benchmark_results.json` so future benchmark changes have a concrete comparison point.
