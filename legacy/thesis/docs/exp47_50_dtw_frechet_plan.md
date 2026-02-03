# EXP47-EXP55 UTM-Based Final Experiment Plan

## Goal
Redo clustering with UTM coordinates (meters) and n=40 trajectories, keeping
experiment numbering sequential after EXP46. Includes a KMeans baseline, best
Euclidean OPTICS variants, HDBSCAN baselines, and DTW/Frechet OPTICS runs.

## Dataset
- Preprocessed input: `data/preprocessed/preprocessed_OPTICS_exp_5_lenmed_top80_n40.csv`
- Each flight has exactly 40 points, using `x_utm` and `y_utm`.

## Feature Vector
- Vector columns: `x_utm`, `y_utm`
- Dimensionality: 40 points x 2 = 80 features (flattened)
- Distance metrics: Euclidean, DTW, discrete Frechet

## Algorithm Choice (Compatibility Note)
- DTW/Frechet requires precomputed distances.
- Supported algorithms in this repo: OPTICS / DBSCAN / HDBSCAN.
- KMeans or two-stage KMeans->OPTICS are not compatible with precomputed distances.

## Final Experiments (EXP47-EXP55)
Parameters are based on the best Euclidean OPTICS settings from EXP40/EXP46 and
the original baselines from EXP01-EXP31.

1) EXP47_kmeans_k6_baseline_utm
   - metric: Euclidean
   - KMeans: k=6

2) EXP48_optics_euclid_ms8_xi03_mcs05_utm
   - metric: Euclidean
   - OPTICS: min_samples=8, xi=0.03, min_cluster_size=0.05

3) EXP49_optics_euclid_ms8_xi03_mcs06_utm
   - metric: Euclidean
   - OPTICS: min_samples=8, xi=0.03, min_cluster_size=0.06

4) EXP50_optics_euclid_ms8_xi02_mcs03_utm
   - metric: Euclidean
   - OPTICS: min_samples=8, xi=0.02, min_cluster_size=0.03

5) EXP51_hdbscan_default_utm
   - metric: Euclidean
   - HDBSCAN: min_cluster_size=60, min_samples=12 (eom)

6) EXP52_hdbscan_large_clusters_utm
   - metric: Euclidean
   - HDBSCAN: min_cluster_size=120, min_samples=12

7) EXP53_kmeans6_optics_refine_utm
   - metric: Euclidean
   - Two-stage: KMeans k=6 -> OPTICS

8) EXP54_optics_dtw_ms8_xi03_mcs05_utm
   - metric: DTW
   - OPTICS: min_samples=8, xi=0.03, min_cluster_size=0.05

9) EXP55_optics_frechet_ms8_xi03_mcs05_utm
   - metric: Frechet
   - OPTICS: min_samples=8, xi=0.03, min_cluster_size=0.05

## Config File
Updated grid file: `experiments/named_experiments.yaml`
- EXP38-EXP46 moved to `completed_experiments`.
- EXP47-EXP55 added under `experiments`.
- Uses the n=40 preprocessed CSV.

## Cloud Job
Updated Slurm job file: `jobs/run_experiment_grid.job`
- Runs `scripts/run_experiment_grid.py --grid experiments/named_experiments.yaml`

## Cloud Transfer Checklist
These are ignored by git and must be copied manually:
- `data/preprocessed/preprocessed_OPTICS_exp_5_lenmed_top80_n40.csv`
- (Optional for provenance) `data/preprocessed/preprocessed_OPTICS_exp_5_lenmed_top80_n40.meta.json`
- (Optional) `data/preprocessed/preprocessed_OPTICS_exp_5_lenmed_top80_n40.removed.csv`

Required repo files (already in git):
- `experiments/named_experiments.yaml`
- `jobs/run_experiment_grid.job`
- `scripts/run_experiment_grid.py`
- `config/backbone_full.yaml`
- `clustering/`, `distance_metrics.py`, and dependencies in `requirements.txt`

## Expected Outputs
For each EXP47-EXP55 experiment:
- `output/<experiment_name>/metrics_by_flow.csv`
- `output/<experiment_name>/metrics_global.csv`
- `output/experiments/<experiment_name>/labels_<flow>.csv`
- `output/<experiment_name>/config_resolved.yaml`
