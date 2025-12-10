"""Trajectory smoothing and resampling for backbone clustering.

Handles per-flight smoothing (Savitzky–Golay or moving average fallback) and
interpolation to fixed-length trajectories in either lat/lon or UTM space.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

try:  # Savitzky-Golay is preferred; fall back to moving average if unavailable.
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - defensive
    savgol_filter = None


SMOOTH_DEFAULT_COLUMNS = ["latitude", "longitude", "altitude"]


def smooth_series(values: pd.Series, window_length: int, polyorder: int) -> pd.Series:
    """Smooth a numeric series with Savitzky-Golay or a moving average fallback."""

    if window_length < 3 or len(values) < window_length:
        return values
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        return values

    if savgol_filter:
        poly = min(polyorder, window_length - 1)
        return pd.Series(savgol_filter(values, window_length=window_length, polyorder=poly), index=values.index)

    return values.rolling(window=window_length, center=True, min_periods=1).mean()


def resample_trajectory(
    df: pd.DataFrame,
    n_points: int,
    method: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Resample a trajectory to exactly n_points using time or index as the domain."""

    if method not in {"time", "index"}:
        raise ValueError(f"Unsupported resampling method: {method}")

    traj = df.copy()
    if method == "time":
        t = (traj["timestamp"] - traj["timestamp"].min()).dt.total_seconds()
    else:
        t = np.linspace(0.0, 1.0, len(traj))

    target = np.linspace(t.min(), t.max(), n_points)
    resampled: Dict[str, Iterable[float]] = {"step": np.arange(n_points, dtype=int)}

    for col in columns:
        if col in traj.columns:
            resampled[col] = np.interp(target, t, traj[col].astype(float))

    meta_cols = ["A/D", "Runway", "flight_id", "icao24", "callsign", "aircraft_type_match"]
    for col in meta_cols:
        if col in traj.columns:
            resampled[col] = traj[col].iloc[0]

    return pd.DataFrame(resampled)


def preprocess_flights(
    df: pd.DataFrame,
    smoothing_cfg: Dict[str, object],
    resampling_cfg: Dict[str, object],
    use_utm: bool = False,
) -> pd.DataFrame:
    """
    For each (flight_id, flow), sort by timestamp, optionally smooth, and resample to fixed points.
    Returns concatenated resampled trajectories.
    """

    smooth_enabled = smoothing_cfg.get("enabled", True)
    window_length = int(smoothing_cfg.get("window_length", 7))
    polyorder = int(smoothing_cfg.get("polyorder", 2))
    smooth_cols = list(smoothing_cfg.get("columns", SMOOTH_DEFAULT_COLUMNS))
    if use_utm:
        smooth_cols = list({*smooth_cols, "x_utm", "y_utm"})

    n_points = int(resampling_cfg.get("n_points", 40))
    method = str(resampling_cfg.get("method", "time")).lower()
    resample_cols = ["altitude", "dist_to_airport_m"]
    if use_utm:
        resample_cols = ["x_utm", "y_utm", *resample_cols, "latitude", "longitude"]
    else:
        resample_cols = ["latitude", "longitude", *resample_cols]

    outputs: List[pd.DataFrame] = []
    for (_, _, flight_id), flight in df.groupby(["A/D", "Runway", "flight_id"]):
        flight_sorted = flight.sort_values("timestamp")
        if len(flight_sorted) < 2:
            continue

        if smooth_enabled:
            for col in smooth_cols:
                if col in flight_sorted.columns:
                    flight_sorted[col] = smooth_series(flight_sorted[col].astype(float), window_length, polyorder)

        try:
            resampled = resample_trajectory(
                flight_sorted,
                n_points=n_points,
                method=method,
                columns=resample_cols,
            )
            outputs.append(resampled)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Skipping flight %s due to resampling error: %s", flight_id, exc)
            continue

    if not outputs:
        return pd.DataFrame()

    result = pd.concat(outputs, ignore_index=True)
    return result
