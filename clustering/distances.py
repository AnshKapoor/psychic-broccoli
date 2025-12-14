"""Feature and distance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from distance_metrics import dtw_distance, discrete_frechet_distance


def build_feature_matrix(
    flights_df,
    vector_cols: Sequence[str],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Returns (X, trajectories) where X is (n_flights, n_features).
    Expects flights_df to have columns: step, flight_id, and vector_cols per step.
    """

    feature_rows: List[List[float]] = []
    trajs: List[np.ndarray] = []
    for _, flight in flights_df.groupby("flight_id"):
        flight_sorted = flight.sort_values("step")
        vec: List[float] = []
        traj_coords: List[Tuple[float, float]] = []
        for _, row in flight_sorted.iterrows():
            coords = [float(row[col]) for col in vector_cols if col in row]
            vec.extend(coords)
            if len(coords) >= 2:
                traj_coords.append((coords[0], coords[1]))
        feature_rows.append(vec)
        trajs.append(np.array(traj_coords, dtype=float))
    X = np.array(feature_rows, dtype=float)
    return X, trajs


def _hash_config(flow_name: str, metric: str, params: dict, flight_ids: Iterable) -> str:
    payload = json.dumps(
        {"flow": flow_name, "metric": metric, "params": params, "flight_ids": list(flight_ids)}, sort_keys=True
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def pairwise_distance_matrix(
    trajectories: Sequence[np.ndarray],
    metric: str = "euclidean",
    cache_dir: Path | None = None,
    flow_name: str | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """
    Compute symmetric pairwise distance matrix.
    Supported metrics: euclidean, dtw, frechet.
    """

    metric = metric.lower()
    params = params or {}
    if cache_dir and flow_name:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _hash_config(flow_name, metric, params, range(len(trajectories)))
        cache_path = cache_dir / f"dist_{key}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    n = len(trajectories)
    D = np.zeros((n, n), dtype=float)

    if metric == "euclidean":
        # For euclidean, expect flattened vectors passed instead of trajectories.
        flat = np.array(trajectories, dtype=float)
        D = cdist(flat, flat, metric="euclidean")
    elif metric in {"dtw", "frechet"}:
        for i in range(n):
            D[i, i] = 0.0
            for j in range(i + 1, n):
                if metric == "dtw":
                    d = dtw_distance(trajectories[i], trajectories[j])
                else:
                    d = discrete_frechet_distance(trajectories[i], trajectories[j])
                D[i, j] = D[j, i] = d
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")

    if cache_dir and flow_name:
        np.save(cache_dir / f"dist_{key}.npy", D)
    return D
