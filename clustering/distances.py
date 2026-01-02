"""Feature and distance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import logging
import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from distance_metrics import dtw_distance, discrete_frechet_distance

logger = logging.getLogger(__name__)


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
    trajectories: Sequence[np.ndarray] | np.ndarray,
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

    if metric == "euclidean":
        # Accept either a 2D feature matrix, or a sequence of arrays (flattened).
        if isinstance(trajectories, np.ndarray):
            if trajectories.ndim != 2:
                raise ValueError("Euclidean distance requires a 2D feature matrix.")
            flat = trajectories.astype(float, copy=False)
        else:
            flat = np.stack([np.asarray(t, dtype=float).ravel() for t in trajectories], axis=0)
        D = cdist(flat, flat, metric="euclidean")
    elif metric in {"dtw", "frechet"}:
        if isinstance(trajectories, np.ndarray):
            raise ValueError(f"{metric} distance requires a sequence of (T,D) trajectories.")
        n = len(trajectories)
        D = np.zeros((n, n), dtype=float)
        total_pairs = n * (n - 1) // 2
        dtw_window_size = params.get("dtw_window_size") if params else None
        log_every = params.get("log_every") if params else None
        log_every = int(log_every) if log_every else None

        logger.info(
            "Computing %s distance matrix for %d trajectories (%d pairs).",
            metric,
            n,
            total_pairs,
        )
        if metric == "dtw" and dtw_window_size is not None:
            logger.info("DTW window size: %s", dtw_window_size)
        start = time.perf_counter()
        pairs_done = 0
        for i in range(n):
            D[i, i] = 0.0
            for j in range(i + 1, n):
                if metric == "dtw":
                    d = dtw_distance(trajectories[i], trajectories[j], window_size=dtw_window_size)
                else:
                    d = discrete_frechet_distance(trajectories[i], trajectories[j])
                D[i, j] = D[j, i] = d
                pairs_done += 1
                if log_every and pairs_done % log_every == 0:
                    elapsed = time.perf_counter() - start
                    rate = pairs_done / elapsed if elapsed > 0 else 0.0
                    remaining = total_pairs - pairs_done
                    eta = remaining / rate if rate > 0 else float("inf")
                    logger.info(
                        "Distance progress: %d/%d pairs (%.1f%%), %.2f pairs/s, ETA %.1fs",
                        pairs_done,
                        total_pairs,
                        100.0 * pairs_done / total_pairs if total_pairs else 100.0,
                        rate,
                        eta,
                    )
        logger.info("Computed %s distance matrix in %.1fs.", metric, time.perf_counter() - start)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")

    if cache_dir and flow_name:
        np.save(cache_dir / f"dist_{key}.npy", D)
    return D
