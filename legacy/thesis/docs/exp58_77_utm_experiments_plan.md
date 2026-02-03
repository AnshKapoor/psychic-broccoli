# UTM-only Experiment Plan (EXP58-EXP77)

## Scope
Design the next 20 clustering experiments using **UTM coordinates only** and the corrected preprocessed file. This grid follows the strongest algorithm families from EXP01-45, with parameter sweeps aimed at reducing noise/over-fragmentation while keeping separation quality high.

## Inputs (fixed for this grid)
- **Preprocessed CSV:** `data/preprocessed/preprocessed_OPTICS_exp_5_lenmed_top80_n40.csv`
- **Vector columns:** `x_utm`, `y_utm`
- **Flows:** `A/D - Runway` (same include list as current named experiments)
- **Resampling n_points (for logging):** 40
- **Logs:** each run writes `output/<EXP>/experiment_log.txt`

## EDA snapshot (from `output/eda/utm_n40/summary.json`)
- Total flights: **32,274**
- Length (m): q05 - **9,222**, q95 - **13,777**
- Length/Displacement ratio (q95): **~1.16**
- Outlier count (length or ratio): **3,809**

## What worked in EXP01-45 (used to guide this grid)
- **KMeans k=6** gave the best/most stable silhouette among KMeans variants.
- **Agglomerative (ward)** performed well compared to average/complete linkages.
- **Birch (coarser threshold)** was competitive.
- **Spectral (n_neighbors ~15)** outperformed larger neighbor counts.
- **OPTICS** showed strong separation when `min_samples-8` and `xi-0.02-0.03`.
- **DBSCAN/MeanShift** underperformed on the earlier global dataset.

Note: EXP32-45 were flow-based and used different inputs, but the OPTICS settings around `min_samples=8` and `xi-0.02-0.03` remained the best signals to test with UTM.

## Strategy to validate the best preprocessed file / n_points
1. **Candidate files:** create/compare `n_points - {30, 40, 50}` (same length filter).
2. **Pilot flows:** run on **Landung_27R** and **Start_27R** only to save time.
3. **Pilot algorithms:** KMeans k=6 and OPTICS (ms8, xi=0.03, mcs=0.05).
4. **Pick n_points** that maximizes silhouette and reduces noise fraction without exploding cluster counts.
5. Lock that file as the default for the grid.

## Experiment IDs (EXP58-EXP77)
All experiments use **UTM (x_utm, y_utm)** and the fixed preprocessed file.

| ID | Name | Algorithm | Key Parameters | Rationale |
|---|---|---|---|---|
| EXP58 | kmeans_k6_utm | KMeans | k=6, n_init=30 | Best KMeans baseline from EXP01-04 |
| EXP59 | kmeans_k4_utm | KMeans | k=4 | Lower cluster count baseline |
| EXP60 | gmm_full_k6_utm | GMM | k=6, full | Stronger mixture baseline |
| EXP61 | gmm_diag_k6_utm | GMM | k=6, diag | Lower-variance baseline |
| EXP62 | agglom_ward_k6_utm | Agglomerative | k=6, ward | Best linkage in EXP06 |
| EXP63 | agglom_ward_k8_utm | Agglomerative | k=8, ward | Check higher k |
| EXP64 | agglom_average_k6_utm | Agglomerative | k=6, average | Compare to ward |
| EXP65 | birch_thr04_k6_utm | Birch | threshold=0.4, k=6 | Mid threshold sweep |
| EXP66 | birch_thr06_k6_utm | Birch | threshold=0.6, k=6 | Coarser sweep |
| EXP67 | spectral_nn15_k6_utm | Spectral | n_neighbors=15, k=6 | Best from EXP12 |
| EXP68 | spectral_nn25_k6_utm | Spectral | n_neighbors=25, k=6 | Wider graph |
| EXP69 | hdbscan_mcs60_ms10_utm | HDBSCAN | mcs=60, ms=10 | Smaller clusters |
| EXP70 | hdbscan_mcs80_ms15_utm | HDBSCAN | mcs=80, ms=15 | Balanced density |
| EXP71 | hdbscan_mcs120_ms20_utm | HDBSCAN | mcs=120, ms=20 | Large cluster bias |
| EXP72 | optics_ms8_xi02_mcs03_utm | OPTICS | ms=8, xi=0.02, mcs=0.03 | Best small-xi setting |
| EXP73 | optics_ms8_xi03_mcs03_utm | OPTICS | ms=8, xi=0.03, mcs=0.03 | Base OPTICS |
| EXP74 | optics_ms8_xi03_mcs05_utm | OPTICS | ms=8, xi=0.03, mcs=0.05 | More conservative |
| EXP75 | optics_ms10_xi03_mcs03_utm | OPTICS | ms=10, xi=0.03, mcs=0.03 | More density smoothing |
| EXP76 | optics_ms8_xi03_mcs02_utm | OPTICS | ms=8, xi=0.03, mcs=0.02 | More splits |
| EXP77 | optics_ms8_xi03_mcs04_utm | OPTICS | ms=8, xi=0.03, mcs=0.04 | Midpoint between mcs 0.03/0.05 |

## Logging
- Every run writes: `output/<EXP>/experiment_log.txt` with method, params, flow counts, cluster counts.
- Global and flow metrics: `output/<EXP>/metrics_global.csv` and `output/<EXP>/metrics_by_flow.csv`.

## Next steps
1. Run the grid on cloud (`jobs/run_experiment_grid.job`).
2. Collect metrics and identify top 3-5 experiments per method.
3. If OPTICS/HDBSCAN dominate, narrow to a smaller sweep for final reporting.

## Proposal: distance-metric follow-ups (post EXP58-77)
Use the best-performing Euclidean configs from EXP58-77, then branch into
non-Euclidean distances on a reduced scope (top 1-2 flows) to control cost.

Suggested path (after results are in):
- If OPTICS or HDBSCAN looks strongest, rerun their best settings with:
  - DTW (banded + kNN edges) + HDBSCAN
  - Frechet (RDP simplified + kNN edges) + HDBSCAN
- If KMeans/GMM dominate, keep Euclidean for final reporting and only run
  DTW/Frechet as a qualitative check on one flow (e.g., Landung_27R).

Candidate IDs (placeholders once EXP58-77 results are reviewed):
- EXP78_hdbscan_dtw_bestflow
- EXP79_hdbscan_frechet_bestflow
- EXP80_optics_dtw_bestflow (only if dense distances are feasible)

Last updated (UTC): 2026-01-25 18:00:59Z
