# Standardization Changelog

This log captures structural changes made to standardize experiments, outputs, and logs.

## 2026-02-02
- Standardized experiment outputs under `output/experiments/<EXP>/`.
- Standardized EDA outputs under `output/eda/` with logs in `logs/eda/`.
- Standardized preprocessed outputs under `data/preprocessed/preprocessed_<id>.csv`.
- Updated `scripts/save_preprocessed.py` to auto-assign IDs and update `thesis/docs/preprocessed_registry.md`.
- Added `config/preprocess_grid.yaml` + `scripts/run_preprocess_grid.py` for standardized preprocessing variants.
- Updated clustering runner to write labels as CSV and enforce UTM-only vector columns.
- Added `scripts/plot_backbone_tracks.py` for per-flow/cluster backbone plots in lat/lon.
- Removed redundant visualization script `visualize_outputs.py` and its job.
- Simplified `.vscode/launch.json` to core workflows and prompt-based inputs.
