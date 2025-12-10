"""Interchangeable trajectory distance metrics using only NumPy.

Includes Euclidean, Dynamic Time Warping (DTW), and discrete Fréchet distances,
plus a selector and pairwise distance matrix helper. Trajectories are expected
as numpy arrays shaped (T, D) with D in {2, 3}.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Trajectory = np.ndarray  # shape (T, D), D = 2 or 3
DistanceFn = Callable[[Trajectory, Trajectory], float]


def _validate_trajectories(traj1: Trajectory, traj2: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    """Ensure both trajectories are 2-D NumPy arrays with matching dimensions."""

    if not isinstance(traj1, np.ndarray) or not isinstance(traj2, np.ndarray):
        raise ValueError("Trajectories must be numpy.ndarray instances.")
    if traj1.ndim != 2 or traj2.ndim != 2:
        raise ValueError("Trajectories must be 2-D arrays of shape (T, D).")
    if traj1.shape[1] != traj2.shape[1]:
        raise ValueError(f"Dimensionality mismatch: {traj1.shape[1]} vs {traj2.shape[1]}.")
    if np.isnan(traj1).any() or np.isnan(traj2).any():
        raise ValueError("Trajectories contain NaN values.")
    return traj1.astype(float, copy=False), traj2.astype(float, copy=False)


def euclidean_distance(traj1: Trajectory, traj2: Trajectory) -> float:
    """
    Compute simple Euclidean distance between two trajectories of equal length.

    Distance: || vec(traj1) - vec(traj2) ||_2
    """

    traj1, traj2 = _validate_trajectories(traj1, traj2)
    if traj1.shape != traj2.shape:
        raise ValueError(f"Shape mismatch: {traj1.shape} vs {traj2.shape}.")
    diff = traj1.ravel() - traj2.ravel()
    return float(np.linalg.norm(diff))


def dtw_distance(traj1: Trajectory, traj2: Trajectory, window_size: int | None = None) -> float:
    """Compute DTW distance between two trajectories using dynamic programming."""

    traj1, traj2 = _validate_trajectories(traj1, traj2)
    t1, t2 = traj1.shape[0], traj2.shape[0]
    if window_size is not None and window_size < 0:
        raise ValueError("window_size must be non-negative or None.")

    dp = np.full((t1 + 1, t2 + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, t1 + 1):
        i_idx = i - 1
        j_start = 1
        j_end = t2 + 1
        if window_size is not None:
            j_start = max(1, i - window_size)
            j_end = min(t2 + 1, i + window_size + 1)
        for j in range(j_start, j_end):
            j_idx = j - 1
            cost = float(np.linalg.norm(traj1[i_idx] - traj2[j_idx]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(dp[t1, t2])


def discrete_frechet_distance(traj1: Trajectory, traj2: Trajectory) -> float:
    """Compute discrete Fréchet distance between two trajectories."""

    traj1, traj2 = _validate_trajectories(traj1, traj2)
    t1, t2 = traj1.shape[0], traj2.shape[0]
    ca = np.full((t1, t2), np.inf, dtype=float)

    def d(i: int, j: int) -> float:
        return float(np.linalg.norm(traj1[i] - traj2[j]))

    ca[0, 0] = d(0, 0)
    for i in range(1, t1):
        ca[i, 0] = max(ca[i - 1, 0], d(i, 0))
    for j in range(1, t2):
        ca[0, j] = max(ca[0, j - 1], d(0, j))

    for i in range(1, t1):
        for j in range(1, t2):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d(i, j))

    return float(ca[t1 - 1, t2 - 1])


def get_trajectory_distance_fn(name: str) -> DistanceFn:
    """Return the distance function for a given metric name."""

    normalized = name.lower()
    if normalized == "euclidean":
        return euclidean_distance
    if normalized == "dtw":
        return dtw_distance
    if normalized == "frechet":
        return discrete_frechet_distance
    raise ValueError(f"Unsupported distance metric: {name}")


def pairwise_distance_matrix(trajectories: Sequence[Trajectory], metric: str = "euclidean") -> np.ndarray:
    """Compute a symmetric pairwise distance matrix for the provided trajectories."""

    if not trajectories:
        raise ValueError("No trajectories provided for distance computation.")

    dist_fn = get_trajectory_distance_fn(metric)
    n = len(trajectories)
    mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        mat[i, i] = 0.0
        for j in range(i + 1, n):
            d = dist_fn(trajectories[i], trajectories[j])
            mat[i, j] = mat[j, i] = d
    return mat


if __name__ == "__main__":
    traj1 = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    traj2 = np.array([[0, 0], [1, 2], [2, 4]], dtype=float)

    print("Euclidean:", euclidean_distance(traj1, traj2))
    print("DTW:", dtw_distance(traj1, traj2))
    print("Fréchet:", discrete_frechet_distance(traj1, traj2))
