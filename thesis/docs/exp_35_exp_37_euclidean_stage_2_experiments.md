# Stage 2 Experiments (EXP32+)
## Runway + Arrival/Departure grouping — **EUCLIDEAN-only** distances (UPDATED v2)

This document defines **EXP35–EXP37** using **only Euclidean distance**, ensuring **finite runtime** while still exploring parameter settings that may reveal >1 cluster per `(A/D, Runway)` flow.  
It reflects the observation that many flows collapse to **1 cluster**, which is often a valid outcome for Doc.29 backbone modelling.

---

## Global instructions (apply to every EXP32+ run)

### Grouping (no code changes)
Use per-flow clustering:
```yaml
flows:
  flow_keys: ["A/D", "Runway"]
```

### Ignore HEL runway
Remove `HEL` from `flows.include`.

### Preprocessing (keep fixed for comparability)
- Keep the same smoothing and resampling settings across EXP35–37  
- **Exception:** EXP35 may optionally reduce `n_points` to 40 (see below)

### Metrics to record (per flow AND overall)
Record per `(A/D, Runway)`:
- `n_clusters`
- `noise_frac`
- `silhouette`
- `davies_bouldin`
- `calinski_harabasz`
- `n_flights`
- `wall_time_sec` (if available)

> If `n_clusters < 2`, silhouette / DB / CH are undefined.  
> Keep existing logic: `reason = "<2 clusters"`.

---

# EXP35–EXP37 (EUCLIDEAN REVISED)

## EXP35 — OPTICS (Euclidean) **Split-Hunter**
**Goal:** Maximise sensitivity to secondary route structure while remaining finite-time.

This experiment intentionally uses **aggressive OPTICS parameters** to test whether the observed “1 cluster” outcome is intrinsic or parameter-driven.

### What to run
- **Target flow(s):**  
  Start with `Start, 09R`  
  (Optionally add other visually multi-corridor flows.)
- **Optional flight cap (recommended):**  
  Sample at most **N = 2000 flights per flow** (fixed random seed).

### Parameters
```yaml
clustering:
  method: "optics"
  distance_metric: "euclidean"
  optics:
    min_samples: 6          # very sensitive
    xi: 0.02                # fine-grained split threshold
    min_cluster_size: 0.03  # if supported; else omit
```

### Optional runtime guardrail
If supported, reduce trajectory resolution **only for this experiment**:
```yaml
preprocessing:
  resampling:
    n_points: 40
```

### Interpretation
- If EXP35 still yields `n_clusters = 1`, the flow is likely **geometrically unimodal**.
- Proceed with **1 backbone + lateral dispersion** (Doc.29-consistent).

---

## EXP36 — OPTICS (Euclidean) **Conservative Stability Check**
**Goal:** Verify that any split found in EXP35 is not a parameter artefact.

### What to run
- **Target flow(s):** Same as EXP35 (e.g. `Start, 09R`)

### Parameters
```yaml
clustering:
  method: "optics"
  distance_metric: "euclidean"
  optics:
    min_samples: 10         # more conservative
    xi: 0.05
    min_cluster_size: 0.05  # if supported; else omit
```

### Interpretation
- EXP35 splits + EXP36 collapses → weak / fine-grained structure
- Both split similarly → strong evidence for **true multi-route geometry**

---

## EXP37 — Lightweight Baselines (Sanity Checks)

Because many flows collapse to 1 cluster, EXP37 is split into **two cheap baseline runs** to benchmark how much OPTICS adds beyond simple methods.

---

### EXP37A — KMeans baseline (k = 2)
- **Target flow(s):** ALL `(A/D, Runway)` flows
```yaml
clustering:
  method: "kmeans"
  k: 2
  distance_metric: "euclidean"
```

---

### EXP37B — Agglomerative Ward baseline (k = 2)
- **Target flow(s):** ALL `(A/D, Runway)` flows
```yaml
clustering:
  method: "agglomerative"
  linkage: "ward"
  n_clusters: 2
  distance_metric: "euclidean"
```

### Rationale
If even these baselines:
- collapse to 1 dominant grouping, or
- produce unstable / meaningless splits,

then the flow should be modelled as **1 backbone + dispersion**, not forced into multiple route clusters.

---

## Reporting table (fill after each run)

| Exp# | Name | Flow filter | Distance | Key params | Flow (A/D,Runway) | n_flights | n_clusters | noise_frac | silhouette | davies_bouldin | calinski_harabasz | wall_time_sec | Notes |
|------|------|------------|----------|------------|-------------------|----------:|-----------:|-----------:|-----------:|---------------:|------------------:|--------------:|------|
| 35 | OPTICS_euclid_splitHunter | Start-09R | euclid | ms=6, xi=0.02 | Start-09R | | | | | | | | optional n_points=40 |
| 36 | OPTICS_euclid_conservative | Start-09R | euclid | ms=10, xi=0.05 | Start-09R | | | | | | | | stability check |
| 37A | KMeans_k2_baseline | ALL | euclid | k=2 | Landung-09L | | | | | | | | repeat per flow |
| 37B | AgglomWard_k2_baseline | ALL | euclid | k=2 | Landung-09L | | | | | | | | repeat per flow |

---

## Recommended execution order (to avoid wasted compute)

1. Run **EXP37A + EXP37B** on all flows  
   → Identify flows with any plausible multi-route structure.
2. Run **EXP35** on the best candidate flow(s) (e.g. `Start-09R`).
3. Run **EXP36** only to confirm stability of any split found in EXP35.

If most flows remain `n_clusters = 1`, proceed with **Doc.29 modelling** using:

> **1 backbone track + σ(s) + standard 7 subtracks**

rather than forcing additional clusters.

This outcome is methodologically valid and consistent with the literature.

