# Experiment Plan: EXP001–EXP050

Goal: Define a clean, thesis-ready experiment set that answers all research questions with UTM-only clustering, standardized preprocessing, and consistent logs/outputs.

Assumptions
- Clustering vectors use UTM only (x_utm, y_utm) unless explicitly noted (altitude ablation).
- All runs are flow-based (A/D, Runway) unless explicitly noted.
- Preprocessed datasets are standardized as `data/preprocessed/preprocessed_<id>.csv` and tracked in `preprocessed_registry.md`.
- Noise simulation will be run only for selected “best” candidates.

Preprocessed IDs (expected)
- preprocessed_1: n_points=40, interpolation=time, smoothing=savgol
- preprocessed_2: n_points=60, interpolation=time, smoothing=savgol
- preprocessed_3: n_points=80, interpolation=time, smoothing=savgol
- preprocessed_4: n_points=50, interpolation=distance, smoothing=savgol
- preprocessed_5: n_points=50, interpolation=time, smoothing=none
- preprocessed_6: n_points=50, interpolation=time, smoothing=moving_average
- preprocessed_7: n_points=50, interpolation=time, smoothing=savgol, include_altitude

Experiment Matrix (EXP001–EXP050)
| id | uid | question | preprocessed_id | distance | algorithm | params | notes |
|---|---|---|---|---|---|---|---|
| EXP001 | EXP001_optics_euclidean_preprocessed_1 | RQ1 n_points | preprocessed_1 | euclidean | optics | ms=12, xi=0.04, mcs=0.04 | baseline n_points=40 |
| EXP002 | EXP002_optics_euclidean_preprocessed_2 | RQ1 n_points | preprocessed_2 | euclidean | optics | ms=12, xi=0.04, mcs=0.04 | n_points=60 |
| EXP003 | EXP003_optics_euclidean_preprocessed_3 | RQ1 n_points | preprocessed_3 | euclidean | optics | ms=12, xi=0.04, mcs=0.04 | n_points=80 |

| EXP004 | EXP004_optics_euclidean_preprocessed_4 | RQ5 preprocessing | preprocessed_4 | euclidean | optics | ms=12, xi=0.04, mcs=0.04 | interpolation=distance |
| EXP005 | EXP005_optics_euclidean_preprocessed_5 | RQ5 preprocessing | preprocessed_5 | euclidean | optics | ms=12, xi=0.04, mcs=0.04 | smoothing=none |
| EXP006 | EXP006_optics_euclidean_preprocessed_6 | RQ5 preprocessing | preprocessed_6 | euclidean | optics | ms=12, xi=0.04, mcs=0.04 | smoothing=moving_average |

| EXP007 | EXP007_hdbscan_dtw_preprocessed_1 | RQ2 distance | preprocessed_1 | dtw | hdbscan | mcs=20, ms=10 | DTW (band_ratio=0.08, LB-Keogh) |
| EXP008 | EXP008_hdbscan_frechet_preprocessed_1 | RQ2 distance | preprocessed_1 | frechet | hdbscan | mcs=20, ms=10 | Frechet (band_ratio=0.08, LB-Keogh) |
| EXP009 | EXP009_hdbscan_euclidean_preprocessed_1 | RQ2 distance | preprocessed_1 | euclidean | hdbscan | mcs=20, ms=10 | Euclidean control |

| EXP010 | EXP010_optics_euclidean_preprocessed_1 | RQ3 distance fit | preprocessed_1 | euclidean | optics | ms=8, xi=0.04, mcs=0.04 | OPTICS min_samples low |
| EXP011 | EXP011_optics_euclidean_preprocessed_1 | RQ3 distance fit | preprocessed_1 | euclidean | optics | ms=12, xi=0.04, mcs=0.04 | OPTICS baseline |
| EXP012 | EXP012_optics_euclidean_preprocessed_1 | RQ3 distance fit | preprocessed_1 | euclidean | optics | ms=16, xi=0.04, mcs=0.04 | OPTICS min_samples high |

| EXP013 | EXP013_optics_euclidean_preprocessed_1 | RQ3 distance fit | preprocessed_1 | euclidean | optics | ms=12, xi=0.03, mcs=0.04 | OPTICS xi low |
| EXP014 | EXP014_optics_euclidean_preprocessed_1 | RQ3 distance fit | preprocessed_1 | euclidean | optics | ms=12, xi=0.06, mcs=0.04 | OPTICS xi high |
| EXP015 | EXP015_optics_euclidean_preprocessed_1 | RQ3 distance fit | preprocessed_1 | euclidean | optics | ms=12, xi=0.04, mcs=0.03 | OPTICS min_cluster_size low |
| EXP016 | EXP016_optics_euclidean_preprocessed_1 | RQ3 distance fit | preprocessed_1 | euclidean | optics | ms=12, xi=0.04, mcs=0.06 | OPTICS min_cluster_size high |

| EXP017 | EXP017_kmeans_euclidean_preprocessed_1 | RQ4 algorithm | preprocessed_1 | euclidean | kmeans | k=4 | baseline kmeans |
| EXP018 | EXP018_kmeans_euclidean_preprocessed_1 | RQ4 algorithm | preprocessed_1 | euclidean | kmeans | k=6 | baseline kmeans |
| EXP019 | EXP019_agglomerative_euclidean_preprocessed_1 | RQ4 algorithm | preprocessed_1 | euclidean | agglomerative | linkage=ward, k=6 | hierarchical |
| EXP020 | EXP020_agglomerative_euclidean_preprocessed_1 | RQ4 algorithm | preprocessed_1 | euclidean | agglomerative | linkage=average, k=6 | hierarchical |

| EXP021 | EXP021_hdbscan_euclidean_preprocessed_1 | RQ4 algorithm | preprocessed_1 | euclidean | hdbscan | mcs=20, ms=10 | density baseline |
| EXP022 | EXP022_dbscan_euclidean_preprocessed_1 | RQ4 algorithm | preprocessed_1 | euclidean | dbscan | eps=250, ms=12 | density baseline |

| EXP023 | EXP023_hdbscan_euclidean_preprocessed_4 | RQ6 interpolation | preprocessed_4 | euclidean | hdbscan | mcs=20, ms=10 | distance interpolation |
| EXP024 | EXP024_hdbscan_euclidean_preprocessed_1 | RQ6 interpolation | preprocessed_1 | euclidean | hdbscan | mcs=20, ms=10 | time interpolation control |

| EXP025 | EXP025_optics_euclidean_preprocessed_7 | RQ7 altitude | preprocessed_7 | euclidean | optics | ms=12, xi=0.04, mcs=0.04 | 3D (x,y,alt) |
| EXP026 | EXP026_hdbscan_euclidean_preprocessed_7 | RQ7 altitude | preprocessed_7 | euclidean | hdbscan | mcs=20, ms=10 | 3D (x,y,alt) |

| EXP027 | EXP027_hdbscan_dtw_preprocessed_2 | RQ2 distance | preprocessed_2 | dtw | hdbscan | mcs=20, ms=10 | DTW + n_points=60 |
| EXP028 | EXP028_hdbscan_frechet_preprocessed_2 | RQ2 distance | preprocessed_2 | frechet | hdbscan | mcs=20, ms=10 | Frechet + n_points=60 |

| EXP029 | EXP029_optics_euclidean_preprocessed_2 | RQ5 preprocessing | preprocessed_2 | euclidean | optics | ms=12, xi=0.04, mcs=0.04 | n_points=60 repeat for consistency |
| EXP030 | EXP030_hdbscan_euclidean_preprocessed_3 | RQ5 preprocessing | preprocessed_3 | euclidean | hdbscan | mcs=20, ms=10 | n_points=80 with HDBSCAN |

| EXP031 | EXP031_birch_euclidean_preprocessed_1 | RQ4 algorithm | preprocessed_1 | euclidean | birch | threshold=0.3, k=6 | optional algorithm |
| EXP032 | EXP032_gmm_euclidean_preprocessed_1 | RQ4 algorithm | preprocessed_1 | euclidean | gmm | n_components=6, cov=full | optional algorithm |

| EXP033 | EXP033_kmeans_euclidean_preprocessed_1 | RQ1 n_points | preprocessed_1 | euclidean | kmeans | k=6 | n_points=40 |
| EXP034 | EXP034_kmeans_euclidean_preprocessed_2 | RQ1 n_points | preprocessed_2 | euclidean | kmeans | k=6 | n_points=60 |
| EXP035 | EXP035_kmeans_euclidean_preprocessed_3 | RQ1 n_points | preprocessed_3 | euclidean | kmeans | k=6 | n_points=80 |

| EXP036 | EXP036_hdbscan_euclidean_preprocessed_1 | RQ1 n_points | preprocessed_1 | euclidean | hdbscan | mcs=20, ms=10 | n_points=40 |
| EXP037 | EXP037_hdbscan_euclidean_preprocessed_2 | RQ1 n_points | preprocessed_2 | euclidean | hdbscan | mcs=20, ms=10 | n_points=60 |
| EXP038 | EXP038_hdbscan_euclidean_preprocessed_3 | RQ1 n_points | preprocessed_3 | euclidean | hdbscan | mcs=20, ms=10 | n_points=80 |

| EXP039 | EXP039_hdbscan_euclidean_preprocessed_5 | RQ5 preprocessing | preprocessed_5 | euclidean | hdbscan | mcs=20, ms=10 | smoothing=none |
| EXP040 | EXP040_hdbscan_euclidean_preprocessed_6 | RQ5 preprocessing | preprocessed_6 | euclidean | hdbscan | mcs=20, ms=10 | smoothing=moving_average |

| EXP041 | EXP041_optics_euclidean_preprocessed_4 | RQ6 interpolation | preprocessed_4 | euclidean | optics | ms=12, xi=0.04, mcs=0.04 | distance interpolation |
| EXP042 | EXP042_kmeans_euclidean_preprocessed_4 | RQ6 interpolation | preprocessed_4 | euclidean | kmeans | k=6 | distance interpolation |

| EXP043 | EXP043_optics_dtw_preprocessed_1 | RQ2 distance | preprocessed_1 | dtw | optics | ms=12, xi=0.04, mcs=0.04 | DTW+OPTICS (check suitability) |
| EXP044 | EXP044_optics_frechet_preprocessed_1 | RQ2 distance | preprocessed_1 | frechet | optics | ms=12, xi=0.04, mcs=0.04 | Frechet+OPTICS |

| EXP045 | EXP045_kmeans_euclidean_preprocessed_7 | RQ7 altitude | preprocessed_7 | euclidean | kmeans | k=6 | 3D (x,y,alt) |
| EXP046 | EXP046_dbscan_euclidean_preprocessed_7 | RQ7 altitude | preprocessed_7 | euclidean | dbscan | eps=250, ms=12 | 3D (x,y,alt) |

| EXP047 | EXP047_optics_euclidean_preprocessed_1 | RQ4 algorithm | preprocessed_1 | euclidean | optics | ms=12, xi=0.04, mcs=0.04 | candidate for noise sim |
| EXP048 | EXP048_hdbscan_euclidean_preprocessed_1 | RQ4 algorithm | preprocessed_1 | euclidean | hdbscan | mcs=20, ms=10 | candidate for noise sim |
| EXP049 | EXP049_kmeans_euclidean_preprocessed_1 | RQ4 algorithm | preprocessed_1 | euclidean | kmeans | k=6 | candidate for noise sim |
| EXP050 | EXP050_agglomerative_euclidean_preprocessed_1 | RQ4 algorithm | preprocessed_1 | euclidean | agglomerative | linkage=ward, k=6 | candidate for noise sim |

Notes
- For DTW/Frechet runs, use sparse kNN caching and LB-Keogh (already implemented).
- Noise simulation should be run only on the top candidate(s) from EXP047–EXP050.
- If any run fails or is too slow, record in `experiments_registry.md` and adjust parameters.