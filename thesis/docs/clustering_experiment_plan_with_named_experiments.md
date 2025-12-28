# Clustering Experiment Plan (60 points per trajectory)

This document defines **named experiments** so each run can be stored in a clean, interpretable folder structure.

**Trajectory representation:** each flight is resampled to **60 points per trajectory** before clustering.  
Suggested folder pattern:

```
results/
  EXP01_kmeans_lowk/
  EXP02_kmeans_sweetspot/
  EXP20_optics_moderate_xi/
  ...
```

**Reusable random_state set (where applicable):** `{11, 23, 37, 59, 83}`

**Standard result fields (fill later):**  
`n_clusters | noise_frac | silhouette | davies_bouldin | calinski_harabasz | runtime | stability_notes`

---

## Phase A — Centroid-based baselines

| Exp ID | Experiment Name | Algorithm | Parameter Values | Comment |
|---:|---|---|---|---|
| 01 | kmeans_lowk | K-means | `k=4, n_init=30, max_iter=300, random_state=11` | coarse flow separation |
| 02 | kmeans_sweetspot | K-means | `k=6, n_init=30, random_state=11` | common baseline |
| 03 | kmeans_fine | K-means | `k=8, n_init=30, random_state=11` | finer splits |
| 04 | kmeans_seed_stability | K-means | Exp02 repeated with `random_state ∈ {11,23,37,59,83}` | stability vs seed |
| 05 | minibatch_kmeans | MiniBatchKMeans | `k=6, batch_size=2048, n_init=10, random_state=11` | speed vs quality |

---

## Phase B — Hierarchical clustering

| Exp ID | Experiment Name | Algorithm | Parameter Values | Comment |
|---:|---|---|---|---|
| 06 | agglom_ward_6 | Agglomerative | `n_clusters=6, linkage=ward, metric=euclidean` | strong vector baseline |
| 07 | agglom_complete_6 | Agglomerative | `n_clusters=6, linkage=complete` | compact clusters |
| 08 | agglom_average_6 | Agglomerative | `n_clusters=6, linkage=average` | compromise linkage |

---

## Phase C — Probabilistic clustering

| Exp ID | Experiment Name | Algorithm | Parameter Values | Comment |
|---:|---|---|---|---|
| 09 | gmm_full_6 | Gaussian Mixture | `n_components=6, cov=full, reg_covar=1e-6, random_state=11` | soft assignment |
| 10 | gmm_diag_6 | Gaussian Mixture | `n_components=6, cov=diag, random_state=11` | faster, simpler |
| 11 | gmm_bic_sweep | Gaussian Mixture | `n_components={4,6,8,10}, cov=full, random_state=11` | select via BIC/AIC |

---

## Phase D — Graph-based & incremental methods

| Exp ID | Experiment Name | Algorithm | Parameter Values | Comment |
|---:|---|---|---|---|
| 12 | spectral_nn15 | Spectral | `k=6, affinity=nn, n_neighbors=15, random_state=11` | non-convex flows |
| 13 | spectral_nn30 | Spectral | `k=6, affinity=nn, n_neighbors=30, random_state=11` | neighbor sensitivity |
| 14 | birch_coarse | BIRCH | `threshold=0.5, bf=50, n_clusters=6` | fast baseline |
| 15 | birch_fine | BIRCH | `threshold=0.3, bf=50, n_clusters=6` | finer subclusters |

---

## Phase E — Density-based clustering

| Exp ID | Experiment Name | Algorithm | Parameter Values | Comment |
|---:|---|---|---|---|
| 16 | dbscan_eps030 | DBSCAN | `eps=0.30, min_samples=10` | dense-core baseline |
| 17 | dbscan_eps040 | DBSCAN | `eps=0.40, min_samples=10` | eps sensitivity |
| 18 | dbscan_dense | DBSCAN | `eps=0.40, min_samples=16` | higher density |
| 19 | optics_fine_xi | OPTICS | `min_samples=12, xi=0.05, min_cluster_size=0.05` | more splits |
| 20 | optics_moderate_xi | OPTICS | `min_samples=12, xi=0.07, min_cluster_size=0.05` | balanced |
| 21 | optics_coarse_xi | OPTICS | `min_samples=12, xi=0.10, min_cluster_size=0.05` | conservative |
| 22 | optics_low_density | OPTICS | `min_samples=8, xi=0.07` | more structure |
| 23 | optics_radius_capped | OPTICS | `min_samples=12, xi=0.07, max_eps={0.5,1.0}` | prevent chaining |

---

## Phase F — Adaptive density (recommended)

| Exp ID | Experiment Name | Algorithm | Parameter Values | Comment |
|---:|---|---|---|---|
| 24 | hdbscan_default | HDBSCAN | `min_cluster_size=60, min_samples=12, method=eom` | strong default |
| 25 | hdbscan_large_clusters | HDBSCAN | `min_cluster_size=120, min_samples=12` | fewer clusters |
| 26 | hdbscan_low_density | HDBSCAN | `min_cluster_size=60, min_samples=6` | finer splits |
| 27 | hdbscan_leaf | HDBSCAN | `min_cluster_size=60, min_samples=12, method=leaf` | granular |

---

## Phase G — No-k methods (exploratory)

| Exp ID | Experiment Name | Algorithm | Parameter Values | Comment |
|---:|---|---|---|---|
| 28 | meanshift_quantile02 | MeanShift | `bandwidth=quantile(0.2), bin_seeding=True` | expensive |
| 29 | affinityprop_default | Affinity Propagation | `damping=0.8, pref=median(sim)` | unstable cluster count |

---

## Phase H — Two-stage (coarse → fine)

| Exp ID | Experiment Name | Algorithm | Parameter Values | Comment |
|---:|---|---|---|---|
| 30 | kmeans6_optics_refine | K-means → OPTICS | Stage1: Exp02; Stage2: Exp20 per cluster | structured flows |
| 31 | gmm6_optics_refine | GMM → OPTICS | Stage1: Exp09; Stage2: Exp20 per component | soft→density |

---

## Phase I — Final validation

| Exp ID | Experiment Name | Algorithm | Parameter Values | Comment |
|---:|---|---|---|---|
| 32 | winner_bootstrap | Best 1–2 methods | 80% bootstrap ×5 + seed sweep | robustness check |

---

## Recommendation for thesis narrative
- **Primary comparison:** Agglomerative vs OPTICS vs HDBSCAN  
- **Final choice justification:** stability + runway separation + downstream backbone quality  
- **Appendix:** full experiment grid (this table)

