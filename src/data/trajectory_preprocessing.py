"""Trajectory preprocessing utilities used by the pipeline layer.

The helpers contained here sanitise raw ADS-B trajectories by removing obvious
sensor spikes, smoothing the remaining signal, and interpolating the flight
path to a common length. This keeps the downstream model training code agnostic
to irregular sampling frequencies and missing data points.
"""

import logging
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

# Configure logging once so diagnostics are emitted during preprocessing.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def detect_anomalies(
    data: np.ndarray,
    window_size: int = 5,
    std_threshold: float = 3.0,
) -> np.ndarray:
    """Return a boolean mask that flags statistical outliers.

    Parameters
    ----------
    data:
        Sequence of numeric samples from a single trajectory attribute.
    window_size:
        Size of the centred rolling window used to compute the mean and
        standard deviation.
    std_threshold:
        Number of standard deviations beyond which a point is considered an
        outlier.

    Returns
    -------
    np.ndarray
        Boolean mask where ``True`` marks samples that fall inside the accepted
        statistical range and ``False`` identifies anomalies.
    """

    series = pd.Series(data)
    rolling_mean = series.rolling(window=window_size, center=True).mean()
    rolling_std = series.rolling(window=window_size, center=True).std()

    lower_bound = rolling_mean - (std_threshold * rolling_std)
    upper_bound = rolling_mean + (std_threshold * rolling_std)
    mask = (series >= lower_bound) & (series <= upper_bound)

    # Fill NaN entries—which typically occur at the edges of the rolling
    # window—with ``True`` so the corresponding observations remain in the
    # trajectory.
    return mask.fillna(True).to_numpy()


def _smooth_numeric_column(column: pd.Series) -> pd.Series:
    """Apply a Savitzky-Golay filter to smooth noisy numeric measurements."""

    window_length = min(11, len(column) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length >= 3:
        polyorder = min(3, window_length - 1)
        smoothed = pd.Series(savgol_filter(column, window_length, polyorder=polyorder), index=column.index)
        return smoothed
    return column


def _interpolate_column(
    values: Iterable[float],
    orig_time: np.ndarray,
    new_time: np.ndarray,
) -> np.ndarray:
    """Create a monotone interpolation of the supplied values."""

    interpolator = PchipInterpolator(orig_time, values)
    return interpolator(new_time)


def preprocess_trajectory(trajectory_df: pd.DataFrame, target_length: int = 20) -> pd.DataFrame:
    """Clean and resample a trajectory to a fixed number of observations.

    Parameters
    ----------
    trajectory_df:
        Raw trajectory points ordered by timestamp.
    target_length:
        Desired number of samples in the processed trajectory.

    Returns
    -------
    pd.DataFrame
        Smoothed and interpolated trajectory table. An empty DataFrame is
        returned when there are insufficient points to perform interpolation.
    """

    numerical_cols = [
        "latitude",
        "longitude",
        "groundspeed",
        "track",
        "altitude",
        "vertical_rate",
        "dist_to_airport_m",
        "geoaltitude",
    ]
    df_cleaned = trajectory_df.copy().reset_index(drop=True)

    # Start with an optimistic mask and progressively remove rows that are
    # flagged as anomalies for any numeric attribute.
    all_valid = np.ones(len(df_cleaned), dtype=bool)
    for col in numerical_cols:
        if col in df_cleaned.columns:
            col_valid = detect_anomalies(df_cleaned[col].to_numpy())
            all_valid = np.logical_and(all_valid, col_valid)

    df_cleaned = df_cleaned[all_valid].reset_index(drop=True)

    if len(df_cleaned) < 3:
        logging.warning("Trajectory too short for interpolation (min. 3 points required).")
        return pd.DataFrame()

    # Smooth every numeric feature individually to suppress high-frequency
    # noise that can arise from ADS-B reception artefacts.
    for col in numerical_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = _smooth_numeric_column(df_cleaned[col])

    orig_time = np.linspace(0, 1, len(df_cleaned))
    new_time = np.linspace(0, 1, target_length)
    result_data: Dict[str, Union[np.ndarray, pd.Series, List[pd.Timestamp]]] = {}

    for col in numerical_cols:
        if col in df_cleaned.columns:
            result_data[col] = _interpolate_column(df_cleaned[col].values, orig_time, new_time)

    # Preserve categorical descriptors by copying the earliest value.
    categorical_cols = ["icao24", "callsign"]
    for col in categorical_cols:
        if col in df_cleaned.columns:
            result_data[col] = df_cleaned[col].iloc[0]

    if "timestamp" in df_cleaned.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df_cleaned["timestamp"]):
                unix_timestamps = df_cleaned["timestamp"].astype("int64") // 10**9
            else:
                unix_timestamps = pd.to_datetime(df_cleaned["timestamp"]).astype("int64") // 10**9

            interpolated_timestamps = _interpolate_column(unix_timestamps.values, orig_time, new_time)
            result_data["timestamp"] = pd.to_datetime(interpolated_timestamps, unit="s")
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.error("Error interpolating timestamps: %s", exc)
            if len(df_cleaned) > 0:
                first_ts = pd.to_datetime(df_cleaned["timestamp"].iloc[0])
                last_ts = pd.to_datetime(df_cleaned["timestamp"].iloc[-1])
                total_seconds = (last_ts - first_ts).total_seconds()
                result_data["timestamp"] = [
                    first_ts + pd.Timedelta(seconds=i * total_seconds / max(target_length - 1, 1))
                    for i in range(target_length)
                ]

    result_df = pd.DataFrame(result_data)

    if "timestamp" in result_df.columns:
        other_columns = [col for col in result_df.columns if col != "timestamp"]
        result_df = result_df[["timestamp", *other_columns]]

    return result_df
