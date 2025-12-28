# Stage 2 Experiments (EXP32+): Runway + Arrival/Departure grouping + multi-metric distances

**Goal:** Build Doc.29-ready route clusters by running the best-performing clustering setups **per operation and runway** (A/D × Runway), while expanding the distance metric beyond Euclidean (**DTW, Fréchet**) and tuning OPTICS parameters.

This stage intentionally **does not require code changes** for A/D and Runway separation: the pipeline already supports grouping through YAML keys.

---

## 0) Global instructions (apply to every EXP32+ run)

### 0.1 Grouping (no code changes)
Update YAML grouping so clustering is performed **within each (A/D, Runway) subset**:

- Set:
  - `flows.flow_keys: ["A/D", "Runway"]`

This ensures you obtain **separate clusters per runway and per arrival/departure**, consistent with ECAC Doc.29 route-level modelling (backbone + dispersion).

### 0.2 Ignore HEL runway (insignificant)
Remove/omit HEL from `flows.include`. For example:

```yaml
flows:
  flow_keys: ["A/D", "Runway"]
  include:
    - ["Landung", "09L"]
    - ["Landung", "09R"]
    - ["Landung", "27L"]
    - ["Landung", "27R"]
    - ["Start",   "09L"]
    - ["Start",   "09R"]
    - ["Start",   "27L"]
    - ["Start",   "27R"]
```

*(Use your dataset’s exact A/D labels; the above matches your “Landung/Start” pattern.)*

### 0.3 Keep preprocessing fixed (comparability)
Hold preprocessing constant for Stage 2 (same smoothing + resampling) so performance differences come from **distance metric + clustering**.

### 0.4 Metrics to record (per (A/D,Runway) AND overall)
For each flow and for the overall dataset, record:

- `n_clusters`
- `noise_frac`
- `silhouette`
- `davies_bouldin`
- `calinski_harabasz`
- `n_flights` (or `total_flights`)

Also record runtime (optional but recommended):
- `wall_time_sec`

---

## 1) Why these base methods were selected (from EXP01–31 results)

From `metrics_quality_exp01_31.csv`, the best-performing candidates for refinement were:

1. **EXP19 OPTICS (fine xi)** – strongest global compactness/separation (very high silhouette, very low DB), but needs per-flow runs and parameter balancing.
2. **EXP30 KMeans→OPTICS** – strong, stable “coarse→fine” partition with good metrics and interpretability.
3. **EXP02 KMeans baseline** – a stable per-flow baseline to compare against density-based clustering.

Stage 2 focuses on **OPTICS as the primary method**, per your preference and the observed metrics.

---

## 2) Stage 2 experiment list (continue numbering after EXP31)

> **Important:** Run *each experiment* with the YAML flow grouping enabled (A/D × Runway) and with **HEL excluded**.

### EXP32 — OPTICS (per-flow) with Euclidean distance (tune min_samples)
**Base:** EXP19/EXP23 style OPTICS  
**Intent:** Establish a strong per-flow OPTICS baseline under Euclidean distance, then test sensitivity to `min_samples`.

YAML changes:
```yaml
clustering:
  method: "optics"
  distance_metric: "euclidean"
  optics:
    # sweep min_samples
    min_samples: 8   # run A
    xi: 0.05
```

Repeat with:
- Run A: `min_samples=8`
- Run B: `min_samples=12`
- Run C: `min_samples=16`

*(Keep `xi=0.05` fixed in EXP32 to isolate min_samples effects.)*

---

### EXP33 — OPTICS (per-flow) with Euclidean distance (tune xi)
**Intent:** Control cluster granularity using OPTICS `xi` once a good `min_samples` is identified (likely 12 from Stage 1).

YAML:
```yaml
clustering:
  method: "optics"
  distance_metric: "euclidean"
  optics:
    min_samples: 12
    # sweep xi
    xi: 0.03   # run A
```

Repeat with:
- Run A: `xi=0.03` (finer)
- Run B: `xi=0.05` (medium)
- Run C: `xi=0.08` (coarser)

---

### EXP34 — OPTICS (per-flow) with DTW distance (route-shape alignment)
**Intent:** Use DTW to compare trajectories by shape with temporal/phase flexibility; should help when flights align poorly in time but follow the same path.

YAML:
```yaml
clustering:
  method: "optics"
  distance_metric: "dtw"
  optics:
    min_samples: 12
    xi: 0.05
```

Notes:
- Expect higher compute time than Euclidean.
- Prefer this for flows where Euclidean splits into many tiny lateral ridges.

---

### EXP35 — OPTICS (per-flow) with Fréchet distance (geometry-aware)
**Intent:** Use Fréchet distance to better capture geometric similarity of paths (often more meaningful for trajectories than pointwise Euclidean).

YAML:
```yaml
clustering:
  method: "optics"
  distance_metric: "frechet"
  optics:
    min_samples: 12
    xi: 0.05
```

Notes:
- Often robust for trajectory geometry, may reduce over-splitting in turns.

---

### EXP36 — Two-stage KMeans→OPTICS (per-flow) with Euclidean
**Base:** EXP30  
**Intent:** Keep interpretability + stability by coarse partitioning, then OPTICS refinement inside each coarse group. Run per-flow to avoid cross-runway mixing.

YAML:
```yaml
clustering:
  method: "two_stage"
  stage1:
    method: "kmeans"
    k: 4
  stage2:
    method: "optics"
    distance_metric: "euclidean"
    optics:
      min_samples: 12
      xi: 0.05
```

*(If your code uses EXP30’s style config rather than a `two_stage` block, replicate EXP30’s config structure but with the per-flow grouping enabled.)*

---

### EXP37 — Per-flow KMeans baseline (Euclidean)
**Base:** EXP02  
**Intent:** Provide a simple reference for cluster counts and separability per runway. Helps demonstrate that improvements come from density adaptivity rather than preprocessing.

YAML:
```yaml
clustering:
  method: "kmeans"
  k: 4
  distance_metric: "euclidean"
```

---

## 3) Reporting table (fill after each run)

Use this table in the markdown report. Record results **per flow** (each (A/D,Runway)) and optionally also an overall aggregate row.

| Exp# | Name | Grouping | Distance | Key params | Flow (A/D,Runway) | n_flights | n_clusters | noise_frac | silhouette | davies_bouldin | calinski_harabasz | Notes |
|------|------|----------|----------|------------|-------------------|----------:|-----------:|-----------:|-----------:|---------------:|------------------:|------|
| 32A  | OPTICS_euclid_ms8  | A/D×RW | euclid | ms=8, xi=0.05  | Landung-09L | | | | | | | |
| 32B  | OPTICS_euclid_ms12 | A/D×RW | euclid | ms=12, xi=0.05 | Landung-09L | | | | | | | |
| 32C  | OPTICS_euclid_ms16 | A/D×RW | euclid | ms=16, xi=0.05 | Landung-09L | | | | | | | |
| 33A  | OPTICS_euclid_xi03 | A/D×RW | euclid | ms=12, xi=0.03 | Landung-09L | | | | | | | |
| 33B  | OPTICS_euclid_xi05 | A/D×RW | euclid | ms=12, xi=0.05 | Landung-09L | | | | | | | |
| 33C  | OPTICS_euclid_xi08 | A/D×RW | euclid | ms=12, xi=0.08 | Landung-09L | | | | | | | |
| 34   | OPTICS_dtw         | A/D×RW | dtw    | ms=12, xi=0.05 | Landung-09L | | | | | | | |
| 35   | OPTICS_frechet     | A/D×RW | frechet| ms=12, xi=0.05 | Landung-09L | | | | | | | |
| 36   | KMeans→OPTICS      | A/D×RW | euclid | k=4; ms=12; xi=0.05 | Landung-09L | | | | | | | |
| 37   | KMeans_baseline    | A/D×RW | euclid | k=4 | Landung-09L | | | | | | | |

*(Duplicate rows for each flow you run, e.g., Landung-27R, Start-09L, etc.)*

---

## 4) Acceptance criteria (what “good” looks like in Stage 2)

Per flow, prefer configurations that yield:

- **Operationally plausible route count**: typically **~2–6 clusters** per (A/D,Runway) (before any Doc.29 merging)
- **Low noise_frac** (near 0 unless truly outliers)
- **High silhouette** (relative within that flow)
- **Low Davies–Bouldin**
- **Stability** across adjacent parameter settings (e.g., xi 0.03→0.05 shouldn’t completely reshape clusters)

---

## 5) Deliverables after Stage 2

For the best-performing configuration per flow:
- Save the labeled flights file (flight_id → cluster_id)
- Generate lat/lon plots with clusters in different colors
- Prepare to merge “lateral-split” clusters into a smaller set of **Doc.29 routes** (backbone + dispersion)

