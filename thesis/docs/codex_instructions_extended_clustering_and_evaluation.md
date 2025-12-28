# Codex Instructions: Extended Clustering & Evaluation Pipeline

## Context
You are working in a Python repository that clusters ADS‑B flight trajectories to derive backbone and side‑tracks. The pipeline already includes:
- CSV ingestion and time parsing
- Optional LAT/LON → UTM projection
- Trajectory segmentation, smoothing, and resampling
- Clustering (currently OPTICS, KMeans, Agglomerative)

Your task is to **extend the clustering layer** and **add systematic cluster‑quality evaluation** while keeping the existing pipeline intact and YAML‑driven.

---

## Goals

1. **Add new clustering algorithms**
   - DBSCAN (scikit‑learn)
   - HDBSCAN (`hdbscan` library, optional dependency)
   - Keep OPTICS, KMeans, Agglomerative

2. **Add internal clustering quality metrics**
   - Davies–Bouldin index (DB) – lower is better
   - Silhouette score – higher is better
   - Calinski–Harabasz index (CH) – higher is better

3. **Create an experiment runner**
   - Run clustering per flow (e.g. per runway)
   - Compute metrics per flow and globally
   - Save results (labels, metrics, resolved config)

4. **Remain backward compatible**
   - Do not break existing YAML configs
   - Only extend schema where necessary

---

## Architectural Requirements

### 1. `clustering/registry.py`

Implement a unified clustering registry.

**Interface / Protocol**
```python
class Clusterer:
    name: str
    supports_precomputed: bool

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        ...
```

**Clusterer implementations**
- `OpticsClusterer` (existing)
- `DbscanClusterer` (new)
- `HdbscanClusterer` (new)
  - If `import hdbscan` fails, raise a clear error suggesting `pip install hdbscan`
- `KMeansClusterer` (existing)
- `AgglomerativeClusterer` (existing)

Provide:
```python
def get_clusterer(method_name: str) -> Clusterer:
    ...
```

---

### 2. `clustering/distances.py`

Centralize feature and distance handling.

Functions to implement:
```python
def build_feature_matrix(flights_df, vector_cols) -> np.ndarray:
    """
    Returns a 2D feature matrix (n_flights × n_features)
    based on resampled x/y/(alt) vectors.
    """


def pairwise_distance_matrix(traj_list, metric: str) -> np.ndarray:
    """
    Returns an NxN symmetric distance matrix.
    Supported metrics:
    - 'euclidean'
    - 'dtw'
    - 'frechet'
    """
```

Rules:
- For **euclidean**, use `scipy.spatial.distance.cdist` on feature vectors
- For **dtw/frechet**, compute pairwise distances
- Cache distance matrices on disk using a hash of:
  - flow name
  - distance metric
  - resampling/smoothing params
  - ordered flight IDs

---

### 3. `clustering/evaluation.py`

Implement internal cluster‑quality evaluation.

```python
def compute_internal_metrics(
    X_or_D,
    labels,
    metric_mode: Literal["features", "precomputed"],
    include_noise: bool = False
) -> dict:
    ...
```

**Rules & edge cases**
- Noise label is `-1`
- If `include_noise=False`, remove noise before scoring
- If fewer than **2 non‑noise clusters** exist:
  - Set DB, Silhouette, CH to `NaN`
  - Add a `reason` field (e.g. "<2 clusters")

**Metric computation**
- Silhouette:
  - Precomputed distances → `silhouette_score(D, labels, metric="precomputed")`
  - Feature vectors → `silhouette_score(X, labels)`
- Davies–Bouldin, Calinski–Harabasz:
  - Compute on feature vectors only

Return example:
```json
{
  "davies_bouldin": 0.84,
  "silhouette": 0.41,
  "calinski_harabasz": 126.3,
  "n_clusters": 4,
  "noise_frac": 0.12
}
```

---

### 4. `experiments/runner.py`

Main orchestration script.

Responsibilities:
1. Load YAML config
2. Split data by `flows.flow_keys` and `flows.include`
3. For each flow:
   - Build feature matrix `X`
   - Build distance matrix `D` if required
   - Run selected clustering algorithm
   - Compute internal metrics
4. Aggregate results and save outputs

**Outputs**
```
output/<experiment_name>/
├── metrics_by_flow.csv
├── metrics_global.csv
├── labels_<flow>.parquet
├── config_resolved.yaml
└── runtime_log.txt
```

Global metrics should be aggregated using weighted mean/median (weight = number of flights).

---

### 5. YAML Configuration Extensions (Backward Compatible)

Extend the `clustering` section:
```yaml
clustering:
  method: optics | dbscan | hdbscan | kmeans | agglomerative
  distance_metric: euclidean | dtw | frechet

  dbscan:
    eps: 300
    min_samples: 12

  hdbscan:
    min_cluster_size: 15
    min_samples: null

  evaluation:
    enabled: true
    include_noise: false
    save_per_flow: true
```

Do not remove or rename existing fields.

---

## Testing Requirements

Respect existing `testing.enabled` flag.

When enabled:
- Limit total rows and flights per flow
- Run a full end‑to‑end experiment
- Print a compact summary table to console

Add `pytest` tests for:
- Evaluation edge cases (all noise, single cluster)
- Distance matrix symmetry
- Correct clusterer retrieval from registry

---

## Performance Requirements

- Cache DTW/Frechet distance matrices
- Allow `n_jobs` for parallel distance computation (joblib)
- Avoid recomputation when config hash is unchanged

---

## Deliverables

- Production‑ready Python code
- Unit tests (pytest)
- Updated README with:
  - Example YAML for OPTICS, DBSCAN, HDBSCAN
  - Interpretation of DB / Silhouette / CH

---

## Definition of Done

- All clustering methods selectable via YAML
- Metrics computed and saved per flow and globally
- Pipeline runs end‑to‑end in testing mode
- Results reproducible and logged for thesis experiments

