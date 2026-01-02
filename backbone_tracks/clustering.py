"""Clustering utilities for backbone trajectories.

Builds per-flight feature vectors and applies OPTICS or KMeans per flow, adding
cluster labels back to the trajectory rows. Supports UTM-based clustering and
optional trajectory distance metrics (euclidean, dtw, frechet) for OPTICS.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, OPTICS

from distance_metrics import pairwise_distance_matrix


def build_feature_matrix(
    df: pd.DataFrame, n_points: int, use_utm: bool, flow_keys: Tuple[str, ...]
) -> Tuple[np.ndarray, pd.DataFrame, list[np.ndarray]]:
    """
    Build a 2D feature matrix X of shape (n_flights, 2 * n_points) for clustering.
    Feature vector: UTM if enabled, else lat/lon:
    [x0, y0, x1, y1, ..., xN-1, yN-1] or [lat0, lon0, ...].
    Returns (X, metadata_df, trajectories) where metadata_df has one row per
    (A/D, Runway, flight_id) and trajectories is a list of shape (N, 2) arrays.
    """

    feature_rows = []
    metadata_rows = []
    trajectories: list[np.ndarray] = []
    for group_vals, flight in df.groupby([*flow_keys, "flight_id"]):
        flight_sorted = flight.sort_values("step")
        if len(flight_sorted) < n_points:
            continue
        vec: list[float] = []
        traj_coords: list[tuple[float, float]] = []
        for step in range(n_points):
            row = flight_sorted[flight_sorted["step"] == step]
            if row.empty:
                break
            if use_utm and "x_utm" in row.columns and "y_utm" in row.columns:
                x_val, y_val = float(row["x_utm"].iloc[0]), float(row["y_utm"].iloc[0])
                vec.extend([x_val, y_val])
                traj_coords.append((x_val, y_val))
            else:
                lat_val, lon_val = float(row["latitude"].iloc[0]), float(row["longitude"].iloc[0])
                vec.extend([lat_val, lon_val])
                traj_coords.append((lat_val, lon_val))
        if len(vec) == 2 * n_points:
            feature_rows.append(vec)
            meta = {key: val for key, val in zip([*flow_keys, "flight_id"], group_vals)}
            metadata_rows.append(meta)
            trajectories.append(np.array(traj_coords, dtype=float))

    X = np.array(feature_rows, dtype=float)
    metadata_df = pd.DataFrame(metadata_rows)
    return X, metadata_df, trajectories


def _apply_cluster_labels(df_flow: pd.DataFrame, metadata: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Attach cluster labels to every row of the flow dataframe."""

    label_map = dict(zip(metadata["flight_id"], labels))
    df_flow = df_flow.copy()
    df_flow["cluster_id"] = df_flow["flight_id"].map(label_map)
    return df_flow


def cluster_flow(
    df_flow: pd.DataFrame,
    method: str,
    cfg: Dict[str, object],
    n_points: int,
    max_clusters_per_flow: int | None = None,
    use_utm: bool = False,
    flow_keys: Tuple[str, ...] = ("A/D", "Runway"),
) -> pd.DataFrame:
    """
    Cluster a single flow (A/D, Runway).
    Adds 'cluster_id' to each flight (per flight_id) using OPTICS or KMeans.
    """

    if df_flow.empty:
        return df_flow.assign(cluster_id=pd.Series(dtype=int))

    distance_metric = str(cfg.get("distance_metric", "euclidean")).lower()
    X, metadata_df, trajectories = build_feature_matrix(df_flow, n_points, use_utm=use_utm, flow_keys=flow_keys)
    if len(metadata_df) == 0:
        logging.warning("No feature rows for flow; skipping clustering.")
        return df_flow.assign(cluster_id=pd.Series(dtype=int))

    labels: np.ndarray
    method_lower = method.lower()
    if method_lower == "optics":
        optics_cfg = cfg.get("optics", {})
        distance_params = cfg.get("distance_params", {}) or {}
        dtw_window_size = distance_params.get("dtw_window_size")
        log_every = distance_params.get("log_every")
        if distance_metric not in {"euclidean", "dtw", "frechet"}:
            raise ValueError(f"Unsupported distance metric for OPTICS: {distance_metric}")

        if distance_metric in {"dtw", "frechet"}:
            dist_matrix = pairwise_distance_matrix(
                trajectories,
                metric=distance_metric,
                dtw_window_size=dtw_window_size,
                log_every=log_every,
            )
            model = OPTICS(
                min_samples=int(optics_cfg.get("min_samples", 5)),
                xi=float(optics_cfg.get("xi", 0.05)),
                min_cluster_size=optics_cfg.get("min_cluster_size", 0.05),
                metric="precomputed",
            )
            labels = model.fit_predict(dist_matrix)
        else:
            model = OPTICS(
                min_samples=int(optics_cfg.get("min_samples", 5)),
                xi=float(optics_cfg.get("xi", 0.05)),
                min_cluster_size=optics_cfg.get("min_cluster_size", 0.05),
                metric="euclidean",
            )
            labels = model.fit_predict(X)
    elif method_lower == "kmeans":
        if distance_metric != "euclidean":
            raise ValueError("KMeans currently supports only Euclidean distance.")
        kmeans_cfg = cfg.get("kmeans", {})
        model = KMeans(
            n_clusters=int(kmeans_cfg.get("n_clusters", 4)),
            random_state=int(cfg.get("random_state", 42)),
            n_init="auto",
        )
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    if max_clusters_per_flow:
        unique = sorted(set(label for label in labels if label != -1))
        limited = unique[: max_clusters_per_flow]
        remap = {label: idx for idx, label in enumerate(limited)}
        labels = np.array([remap.get(label, -1) for label in labels])

    return _apply_cluster_labels(df_flow, metadata_df, labels)
