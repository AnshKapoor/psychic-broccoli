# Flight Trajectory Clustering and Noise Simulation - Codebase Overview

## 1. Project Purpose and Domain

### Primary Goal
This thesis codebase integrates ADS-B (Automatic Dependent Surveillance-Broadcast) flight trajectory data with aircraft noise measurements to:
- Match flight trajectories with noise observations at the airport level
- Cluster similar flight paths per runway and direction (Arrival/Departure)
- Generate "backbone tracks" representing typical flight paths
- Create inputs for ECAC Doc29 noise simulation models
- Support noise impact assessment and simulation workflows

### Domain
- **Aerospace/Aviation**: Flight trajectory analysis
- **Environmental**: Aircraft noise measurement and simulation
- **Geospatial**: Coordinate transformation (lat/lon to UTM), airport-centric distance calculations
- **Data Science**: Clustering algorithms (OPTICS, KMeans, DBSCAN, HDBSCAN, Agglomerative), trajectory similarity metrics (DTW, Fréchet distance)

---

## 2. High-Level Architecture

```
Data Pipeline Flow:
====================

1. DATA INGESTION & MATCHING
   ├── ADS-B Joblib files → CSV/Parquet conversion
   ├── Noise Excel workbooks
   └── Spatial/temporal matching: src/data/adsb_noise/

2. PREPROCESSING & SEGMENTATION
   ├── Flight boundary detection (time gap, distance, direction rules)
   ├── Trajectory smoothing (Savitzky-Golay, moving average, EWM)
   ├── Coordinate transformation (lat/lon → UTM)
   └── Resampling to fixed length (n_points)
   Module: backbone_tracks/

3. CLUSTERING BY FLOW
   ├── Per-runway (A/D + Runway) or global grouping
   ├── Feature matrix construction (lat/lon or UTM)
   ├── Distance metrics: Euclidean, DTW, Fréchet
   ├── Clustering algorithms: OPTICS, KMeans, DBSCAN, HDBSCAN, Agglomerative
   └── Per-algorithm parameter exploration
   Modules: backbone_tracks/clustering.py, clustering/

4. BACKBONE GENERATION
   ├── Percentile envelopes (p10, p50, p90) per cluster/flow
   ├── Median-track + side-track representation
   └── Flight count per cluster
   Module: backbone_tracks/backbone.py

5. VISUALIZATION & REPORTING
   ├── Cluster plots (lat/lon or UTM space)
   ├── Backbone visualization with uncertainty bands
   ├── Experiment metrics (Davies-Bouldin, Silhouette, Calinski-Harabasz)
   └── Flow-specific composition analysis
   Modules: backbone_tracks/plots.py, scripts/plot_*.py

6. DOC29 SIMULATION INPUT GENERATION
   ├── Export 7-track layouts (center ± offsets)
   ├── Generate Doc29 groundtrack CSV files
   ├── Create flight input CSV with aircraft/runway metadata
   └── Execute noise simulation
   Modules: scripts/doc29_tracks.py, noise_simulation/

7. NOISE SIMULATION & RESULTS
   ├── ECAC Doc29 noise propagation model
   ├── Cumulative noise contours
   └── Visualization of results
   Modules: scripts/plot_noise_results.py
```

---

## 3. Primary Python Modules and Their Purposes

### Core Pipeline Modules

#### **backbone_tracks/** — Main clustering and backbone pipeline

| Module | Purpose |
|--------|---------|
| `backbone.py` | Computes per-cluster percentile envelopes (p10/p50/p90) from clustered trajectories; produces backbone tracks with flight counts |
| `clustering.py` | Builds per-flight feature matrices and applies clustering algorithms (OPTICS, KMeans, etc.) with support for UTM coordinates and trajectory distance metrics |
| `preprocessing.py` | Handles trajectory smoothing (Savitzky-Golay, moving average, median, EWM) and resampling to fixed-length trajectories in lat/lon or UTM space |
| `segmentation.py` | Multi-rule flight boundary detection: time gaps, direction reversals, distance jumps; optional per-flow flight caps for testing |
| `io.py` | CSV loading with glob patterns, required-column validation, UTC coordinate transformation to UTM, CSV export |
| `config.py` | YAML configuration loading and nested value retrieval with defaults |
| `plots.py` | Visualization of backbone tracks with percentile bands, cluster plots, and flight metadata overlays |

#### **src/data/** — Data preparation and ADS-B/noise matching

| Module | Purpose |
|--------|---------|
| `adsb_noise/pipeline.py` | High-level orchestrator for the ADS-B/noise merger workflow |
| `adsb_noise/loader.py` | Loads joblib/CSV ADS-B data and Excel noise workbooks |
| `adsb_noise/matcher.py` | Time-tolerant spatial/temporal matching between flights and noise measurements |
| `adsb_noise/grouper.py` | Groups raw ADS-B points into flight trajectories |
| `adsb_noise/preprocessor.py` | Filters and cleans flight data (altitude, speed, distance to airport) |
| `adsb_noise/exporter.py` | Exports merged data to Parquet and CSV with deduplication |
| `adsb_noise_merge.py` | CLI entry point for ADS-B/noise matching pipeline |
| `trajectory_preprocessing.py` | Detects anomalies (outliers) and smooths trajectory features |
| `data_preparation.py` | Generic data loading and preparation utilities |

#### **clustering/** — Distance metrics and cluster quality evaluation

| Module | Purpose |
|--------|---------|
| `distances.py` | Feature matrix construction from per-flight trajectory data with caching support |
| `evaluation.py` | Cluster quality metrics: Davies-Bouldin, Silhouette, Calinski-Harabasz; supports precomputed distance matrices |
| `registry.py` | Clustering algorithm registry with wrappers for OPTICS, DBSCAN, HDBSCAN, KMeans, Agglomerative, Birch, MeanShift, Spectral, GaussianMixture, AffinityPropagation |

#### **distance_metrics.py** — Trajectory similarity metrics

Pure NumPy implementations of three distance metrics:
- **Euclidean**: Simple L2 norm between flattened trajectory vectors
- **DTW (Dynamic Time Warping)**: Optimal alignment between temporal sequences
- **Fréchet distance**: Hausdorff-like measure for curve similarity
All accept (T, 2) or (T, 3) shaped arrays (2D or 3D trajectories)

#### **experiments/** — Extended experiment runner and grid sweeps

| Module | Purpose |
|--------|---------|
| `runner.py` | Runs extended clustering experiments on preprocessed CSV: loads config, builds feature matrices, applies clustering, computes metrics per flow and globally, exports labels |

### Data-Related Modules

| Module | Purpose |
|--------|---------|
| `merge_adsb_noise.py` | CLI for matching noise Excel to ADS-B joblib; supports spatial buffer, time windows, airport distance filtering |
| `add_noise_runway_columns.py` | Enriches matched trajectory CSVs with A/D and Runway from noise metadata |

### Visualization and Reporting Scripts (in scripts/)

| Script | Purpose |
|--------|---------|
| `save_preprocessed.py` | Saves preprocessed (smoothed, resampled) trajectories to CSV for downstream analysis |
| `plot_experiment_results.py` | Plots cluster metrics and trajectories per experiment output folder |
| `plot_exp_latlon_flows.py` | Lat/lon cluster plots split by selected flow labels |
| `plot_backbone_tracks.py` | Visualizes backbone tracks in lat/lon with arrival scheme options |
| `legacy/scripts/generate_backbone_tracks.py` | Archived legacy backbone generation path |
| `doc29_tracks.py` | Creates Doc29 7-track layouts from preprocessed data; exports groundtrack CSVs |
| `plot_noise_results.py` | Contour/heatmap plots of Doc29 noise simulation results |
| `plot_metrics_doc29.py` | Metrics plots for Doc29 experiments |
| `convert_adsb_joblib_to_csv.py` | Batch conversion of joblib archives to Parquet format |

### Utilities (in src/utils/)

| Module | Purpose |
|--------|---------|
| `joblib_visualization.py` | Loads joblib datasets and generates human-readable previews |
| `joblib_to_table.py` | Conversion utilities for joblib to tabular formats |

---

## 4. Key Entry Points (Main Scripts)

### Primary Entry Points

#### **1. `cli.py`** — Full backbone clustering pipeline
```bash
python cli.py -c config/backbone.yaml
```
- Loads matched trajectory CSVs
- Segments flights by time/distance rules
- Preprocesses (smooths, resamples) trajectories
- Clusters per flow or globally
- Computes backbone tracks
- Outputs: `preprocessed_*.csv`, `clustered_flights_*.csv`, `backbone_tracks_*.csv`, optional plots

#### **2. `e
