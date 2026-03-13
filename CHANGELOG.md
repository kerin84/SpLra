# Changelog

## v1.0.0-paper - 2026-03-13

This release freezes the repository as the official reproducibility artifact for the REF-UNAH paper.

- Refactored the numerical code into `core`, `io_layer`, and `viz` while preserving the legacy `Sparse_Lra.py` API.
- Added unit tests for the core numerical routines and a notebook smoke test for setup/data loading.
- Standardized notebook setup so the repository can be executed locally without hard-coded Colab paths.
- Added locked top-level dependency manifests for the paper artifact and validation tooling.
- Added English project documentation, citation metadata, and CI automation for lint, tests, and notebook smoke checks.
