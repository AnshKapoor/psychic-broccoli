# AI Agent Guide — Flight__Clustering

This file standardizes how AI agents should work in this repository. Keep updates short and factual.

## Project Summary
- Purpose: ADS-B flight trajectory preprocessing, clustering, and noise-simulation evaluation for thesis work.
- Core data flow: matched ADS‑B trajectories → preprocessing → clustering → backbone tracks → noise simulation.

## Key Conventions
- **UTM only for clustering**. Use `x_utm`, `y_utm` for distance metrics.
- **Outputs are standardized**:
  - Experiments: `output/experiments/<EXP>/`
  - EDA: `output/eda/`
  - Preprocessed: `data/preprocessed/preprocessed_<id>.csv`
  - Logs: `logs/experiments/`, `logs/eda/`
- **Labels are CSV** (not parquet).
- **Runways allowed**: `09L`, `09R`, `27L`, `27R` unless explicitly changed.

## Canonical Configs
- Preprocessing base config: `config/backbone_full.yaml`
- Preprocess variants grid: `config/preprocess_grid.yaml`
- Experiment grid: `experiments/experiment_grid.yaml`

## Primary Scripts
- Preprocess (single): `scripts/save_preprocessed.py`
- Preprocess grid: `scripts/run_preprocess_grid.py`
- Run experiments: `scripts/run_experiment_grid.py`
- EDA lengths: `scripts/eda_raw_lengths.py`
- EDA flight counts: `scripts/eda_flight_counts.py`
- EDA combo: `scripts/run_eda_combo.py`
- Plot backbone tracks: `scripts/plot_backbone_tracks.py`
- Plot clustered flows: `scripts/plot_exp_latlon_flows.py`

## Noise Simulation
- Experiment simulation: `noise_simulation/run_doc29_experiment.py`
- Ground truth batching: `noise_simulation/run_doc29_ground_truth.py`
- NPD mapping: `noise_simulation/automation/aircraft_types.py`

## Launch Configs
See `.vscode/launch.json` for standard runs. Names are prefixed to be visually distinct:
- `Grid: ...` for grid runs
- `EDA ...` for analytics
- `Plot: ...` for visualization

## Logs & Registry
- Preprocessed registry: `thesis/docs/preprocessed_registry.md`
- Experiments registry: `thesis/docs/experiments_registry.md`
- Standardization changelog: `CHANGELOG_STANDARDIZATION.md`

## Expectations for Agents
- Keep outputs in the standardized folders.
- Don’t introduce new naming schemes without updating registries.
- Keep changes small and incremental; document major changes in the changelog.
- Avoid adding unused scripts/configs; remove redundancies when safe.
- Add concise docstrings to new/modified files and functions whenever possible,
  including input/output structure (columns, shapes, or key fields).
