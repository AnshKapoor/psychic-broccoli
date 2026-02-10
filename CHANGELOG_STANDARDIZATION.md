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

## 2026-02-03 13:50:19
- Ran raw-length EDA (`scripts/eda_raw_lengths.py`) and selected p25 length threshold.
- Updated preprocessing minimum length filter to 9.35 km (p25) in `config/backbone_full.yaml` and `config/backbone.yaml`.
- Simplified `.vscode/launch.json` to core workflows and prompt-based inputs.
The results were at the following path: Wrote output\eda\raw_lengths\flight_lengths.csv
Wrote output\eda\raw_lengths\length_summary.json

## 2026-02-04 03:29:54
- Added identity-based flight segmentation (split on icao24/callsign).
- Added EDA flight count script with aircraft type lookup via traffic.
- Added EDA combo runner for raw lengths + flight counts.
- Added launch configs for EDA combo, ADS-B monthly EDA, and prompt-based experiment grid runs.
- Updated preprocess grid to skip existing outputs and continue on errors.

## 2026-02-04 04:45:12
- Preprocess grid run: all preprocessed files created successfully except variants 4 and 9.

## 2026-02-05 00:10:53
- Completed full experiment grid (EXP001–EXP050); all preprocessed files available.

## 2026-02-05 00:20:00
- Added optional RDP (Ramer–Douglas–Peucker) simplification in preprocessing (disabled by default).

## 2026-02-05 04:21:00
- Fixed experiment grid override so per-experiment preprocessed files are respected.
- Added flow-aware cluster reporting (`flow_label`) to avoid cross-flow cluster ID confusion.
- Added sample input vector logging and run duration metadata to experiment logs.
- Added KML export (with optional runway placemarks and noise in gray).
- Added aircraft-specific height profiles + spectral class mapping (with fallbacks + missing-profile warnings).
- Added aggregation script to compare cluster subtracks vs ground truth in energy domain.

## 2026-02-06 04:21:00
- Updated Doc29 main experiment runner (`noise_simulation/run_doc29_experiment.py`) to avoid type-folder collisions by using `ICAO__NPD` folder keys instead of NPD-only keys.
- Extended experiment summaries (`summary_mse.csv` + `mse.json`) with `npd_id`, `type_folder`, `npd_table`, `height_profile`, and `spectral_class` for reproducibility.
- Updated Doc29 ground-truth runner (`noise_simulation/run_doc29_ground_truth.py`) with the same `ICAO__NPD` folder strategy to prevent output overwrites when multiple ICAO types share one NPD ID.
- Updated ground-truth runner to generate track groups keyed by ICAO type (while still selecting NPD tables via mapped NPD ID), so profile/spectral mapping remains type-correct.
- Added robust preprocessed input path fallback in ground-truth runner (`data/preprocessed/...` to `output/preprocessed/...`) to prevent FileNotFound errors in standardized runs.

## 2026-02-06 22:42:02
- Added `noise_simulation/compare_experiment_totals.py` for fair post-Doc29 comparison of clustered prediction vs per-category ground truth using energy-domain aggregation.
- New outputs per experiment:
  - `aggregate_totals/category_summary.csv` (category-level MAE/MSE/RMSE + log-energy average cumulative level)
  - `aggregate_totals/category_aligned_receivers.csv` (receiver-level aligned prediction/ground truth)
  - `aggregate_totals/overall_summary.json` (overall MAE/MSE/RMSE + average cumulative level delta)
- Supports both subtrack weighting modes:
  - `weighted`: uses `Nr.day` values already encoded in `Flight_subtracks.csv`
  - `unweighted`: scales by `n_flights / tracks_per_cluster`

## 2026-02-06 23:28:10
- Archived redundant scripts to reduce active surface area:
  - `legacy/scripts/run_clustering.py`
  - `legacy/scripts/generate_backbone_tracks.py`
  - `legacy/scripts/preprocess_dtw_frechet.py`
  - `legacy/scripts/plot_exp_latlon.py`
  - `legacy/scripts/eda_utm.py`
- Archived redundant noise scripts:
  - `legacy/noise_simulation/aggregate_cluster_cumulative.py`
  - `legacy/noise_simulation/interpret_results.py`
- Archived redundant Slurm jobs:
  - `legacy/jobs/run_eda.job`
  - `legacy/jobs/run_experiment_optics.job`
  - `legacy/jobs/run_save_preprocessed.job`
  - `legacy/jobs/run_backbone_clustering_full.job`
- Updated launch config to replace legacy Doc29 aggregate runner with:
  - `Doc29 Compare Totals (Fair)` using `noise_simulation/compare_experiment_totals.py`
- Updated docs (`README.md`, `CODEBASE_OVERVIEW.md`) to reflect active scripts and archived paths.

2026-02-07 18:26:42
- Added Doc29 Slurm jobs:
  - `jobs/run_doc29_ground_truth.job`
  - `jobs/run_doc29_experiment.job`
  - `jobs/run_doc29_compare_totals.job`

2026-02-09 14:19:46
[
L_{\mathrm{eq},w} = 10 \log_{10}!\left(\frac{t_0}{T_0}\sum_{i=1}^{N} g_i \cdot 10^{L_{E,i}/10}\right) + C
]

Where (L_{E,i}) is the single‑event exposure level, (g_i) is the time‑of‑day weight, (T_0) is the reference period, and (t_0) is the time basis used for normalization