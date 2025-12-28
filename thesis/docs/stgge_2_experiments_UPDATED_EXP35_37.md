# Stage 2 Experiments (EXP32+): Runway + Arrival/Departure grouping + multi-metric distances (UPDATED)

This file updates **EXP35–EXP37** to ensure they run in **finite time** and provide **meaningful parameter variation** given your observation that most flows collapse to **1 cluster** under per-(A/D,Runway) OPTICS.

---

## Global instructions (apply to every EXP32+ run)

### Grouping (no code changes)
Use per-flow clustering:
```yaml
flows:
  flow_keys: ["A/D", "Runway"]
```

### Ignore HEL runway
Remove HEL from `flows.include`.

### Keep preprocessing fixed (comparability)
Keep the same smoothing + resampling settings across EXP35–37 unless explicitly stated below.

### Metrics to record (per (A/D,Runway) AND overall)
Record per flow:
- `n_clusters`, `noise_frac`, `silhouette`, `davies_bouldin`, `calinski_harabasz`, `n_flights`
- add `wall_time_sec` if available

> Note: if `n_clusters < 2`, silhouette/DB/CH are undefined. Keep your existing “reason: <2 clusters” behaviour.

---

# EXP35–EXP37 (REVISED)

## EXP35 — OPTICS with Fréchet distance (bounded runtime)
**Problem observed:** DTW-style distances are too slow at full scale.  
**Fix:** Run Fréchet only on the *one flow that actually produced >1 cluster* so far (**Start-09R**), and bound input size.

### What to run
- **Target flow(s):** ONLY `Start, 09R`
- **Optional cap on flights (recommended):** sample at most **N=1500** flights for this experiment (random seed fixed).  
  - If your pipeline already has any `max_flights_per_flow` / `sample_per_flow` option, use it.
  - If it does not, manually filter the input list of flight_ids for this one run (no code changes; just pre-filter input file list if your tooling allows).

### Parameters (different from Euclidean runs)
```yaml
clustering:
  method: "optics"
  distance_metric: "frechet"
  optics:
    min_samples: 8     # smaller than 12 to encourage splits
    xi: 0.05           # medium granularity
```

### Runtime guardrails (choose what your pipeline supports)
- Prefer `n_points` = 40 for this single experiment if you have a simple YAML override (Fréchet gets faster with shorter sequences).
- Keep the rest unchanged.

**Rationale:** This tests whether a geometry-aware metric reduces “lateral ridge” over-splitting (or reveals a more meaningful split) without blowing up runtime.

---

## EXP36 — Two-stage KMeans→OPTICS (finite + interpretable)
**Goal:** Use two-stage only where it is needed (flows with >1 route), and vary parameters to test stability.

### What to run
- **Target flow(s):** ONLY `Start, 09R` initially.
  - If later another flow yields >1 cluster under finer OPTICS, add it.

### Parameters (changed from earlier draft)
Use a smaller K and a finer OPTICS xi to avoid trivial “1 cluster” outcomes.

```yaml
clustering:
  method: "two_stage"
  stage1:
    method: "kmeans"
    k: 3                 # changed from 4
  stage2:
    method: "optics"
    distance_metric: "euclidean"
    optics:
      min_samples: 10    # changed from 12
      xi: 0.03           # finer than 0.05/0.08
```

**Why this is finite-time:** Euclidean distance + restricted flow keeps computation small.

---

## EXP37 — Lightweight baselines (two quick baselines instead of one)
Because many flows collapse to 1 cluster, EXP37 is repurposed into **two cheap runs** to benchmark how much “extra” OPTICS gives.

### EXP37A — Per-flow KMeans baseline (small k)
- **Target flow(s):** ALL flows (A/D×Runway) are OK; this is fast.
```yaml
clustering:
  method: "kmeans"
  k: 2                  # smaller than earlier k=4
  distance_metric: "euclidean"
```

### EXP37B — Per-flow Agglomerative Ward baseline
- **Target flow(s):** ALL flows (A/D×Runway)
```yaml
clustering:
  method: "agglomerative"
  linkage: "ward"
  n_clusters: 2
  distance_metric: "euclidean"
```

**Rationale:** These provide fast “sanity” baselines per flow. If even these baselines yield 1 dominant grouping (or produce unstable splits), it supports the conclusion that the flow is essentially unimodal and should be modelled as **1 backbone + dispersion**.

---

# Reporting table (fill after each run)

| Exp# | Name | Flow filter | Distance | Key params | Flow (A/D,Runway) | n_flights | n_clusters | noise_frac | silhouette | davies_bouldin | calinski_harabasz | wall_time_sec | Notes |
|------|------|------------|----------|------------|-------------------|----------:|-----------:|-----------:|-----------:|---------------:|------------------:|--------------:|------|
| 35   | OPTICS_frechet_ms8_xi05 | Start-09R only | frechet | ms=8, xi=0.05 | Start-09R | | | | | | | | cap flights to <=1500 |
| 36   | KMeans3→OPTICS_ms10_xi03 | Start-09R only | euclid | k=3; ms=10; xi=0.03 | Start-09R | | | | | | | | |
| 37A  | KMeans_k2_baseline | ALL flows | euclid | k=2 | Landung-09L | | | | | | | | duplicate per flow |
| 37B  | AgglomWard_k2_baseline | ALL flows | euclid | k=2 | Landung-09L | | | | | | | | duplicate per flow |

---

# Practical next step logic (so you don’t waste compute)

1. Run **EXP37A/37B** across all flows to see which flows have any plausible multi-route structure.
2. Run **EXP36** on the one flow that clearly splits (**Start-09R**).
3. Run **EXP35** (Fréchet) only if you need to validate that the split is geometric and not an Euclidean artefact.

If most flows remain `n_clusters=1`, proceed to Doc.29 modelling using **1 backbone + σ(s) + 7 subtracks** rather than forcing additional clusters.
