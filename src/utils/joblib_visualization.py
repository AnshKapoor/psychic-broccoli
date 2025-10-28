"""Utilities for inspecting and visualising stored ADS-B joblib datasets."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def load_joblib_dataset(joblib_path: str) -> Any:
    """Return the Python object stored inside a joblib archive.

    Parameters
    ----------
    joblib_path:
        Filesystem path pointing to the ``.joblib`` file to load.

    Returns
    -------
    Any
        Deserialised object contained within ``joblib_path``.
    """

    # Delegate the heavy lifting to joblib while providing a thin wrapper that
    # can be imported by multiple modules without repeating the same call.
    return joblib.load(joblib_path)


def _extract_flight_dataframe(dataset: Any, icao24: str) -> pd.DataFrame:
    """Retrieve a trajectory DataFrame for ``icao24`` from ``dataset``.

    Parameters
    ----------
    dataset:
        Data structure produced by :func:`load_joblib_dataset`. This is expected
        to be compatible with the ``traffic`` library and expose either a
        ``data`` attribute or dictionary-style access.
    icao24:
        Hex-encoded ICAO24 address identifying the flight of interest.

    Returns
    -------
    pd.DataFrame
        Tabular view of the ADS-B observations for the requested aircraft.

    Raises
    ------
    KeyError
        If the desired aircraft is absent from the dataset.
    ValueError
        If the dataset does not provide a DataFrame view compatible with
        plotting.
    """

    flight_entry: Any
    if isinstance(dataset, Dict):
        # Some pipelines might serialise the data as a dictionary keyed by
        # ``icao24`` or callsign values.
        flight_entry = dataset.get(icao24)
        if flight_entry is None:
            raise KeyError(f"Flight {icao24} not found in joblib dataset")
    else:
        try:
            # The ``traffic`` package exposes ``__getitem__`` for ICAO24 lookups.
            flight_entry = dataset[icao24]
        except (KeyError, TypeError) as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Flight {icao24} not found in joblib dataset") from exc

    flight_data: Any = getattr(flight_entry, "data", flight_entry)
    if not isinstance(flight_data, pd.DataFrame):
        raise ValueError("Flight data does not expose a pandas DataFrame")

    return flight_data.reset_index(drop=True)


def plot_joblib_flight(
    joblib_path: str,
    icao24: str,
    *,
    time_window: Optional[Tuple[str, str]] = None,
    title: Optional[str] = None,
) -> pd.DataFrame:
    """Plot the ground track for a single aircraft contained in a joblib file.

    Parameters
    ----------
    joblib_path:
        Path to the ``.joblib`` archive that contains ADS-B trajectory data.
    icao24:
        Hex identifier of the aircraft to visualise.
    time_window:
        Optional tuple describing the start and end timestamps (ISO string
        format) that should be used to filter the trajectory prior to plotting.
    title:
        Optional plot title. When omitted a default title is generated using the
        ``icao24``.

    Returns
    -------
    pd.DataFrame
        DataFrame used for plotting, enabling callers to perform additional
        analysis if desired.
    """

    # Load the dataset and extract the trajectory for the requested aircraft.
    dataset: Any = load_joblib_dataset(joblib_path)
    flight_df: pd.DataFrame = _extract_flight_dataframe(dataset, icao24)

    filtered_df: pd.DataFrame = flight_df
    if time_window is not None:
        start_time_str, end_time_str = time_window
        try:
            # Parse timestamps to guard against invalid input and to support
            # string-based filtering.
            start_time = datetime.fromisoformat(start_time_str)
            end_time = datetime.fromisoformat(end_time_str)
            filtered_df = flight_df[  # type: ignore[index]
                (pd.to_datetime(flight_df["timestamp"]) >= start_time)
                & (pd.to_datetime(flight_df["timestamp"]) <= end_time)
            ].reset_index(drop=True)
        except ValueError:  # pragma: no cover - user input validation
            logger.warning("Invalid ISO timestamps supplied; defaulting to full track")

    if filtered_df.empty:
        logger.warning("No trajectory points available after applying filters")
        return filtered_df

    # Render the trajectory using longitude/latitude to align with the pipeline
    # plots used elsewhere in the project.
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df["longitude"], filtered_df["latitude"], marker="o", linestyle="-")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title or f"Trajectory for {icao24}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return filtered_df


__all__ = ["load_joblib_dataset", "plot_joblib_flight"]
