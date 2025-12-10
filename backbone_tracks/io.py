"""Input/output helpers for the backbone tracks pipeline.

Covers CSV loading with truncation, required-column checks, coordinate
conversion to UTM, and CSV saving.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from pyproj import Transformer

REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "latitude",
    "longitude",
    "altitude",
    "dist_to_airport_m",
    "A/D",
    "Runway",
]


def load_monthly_csvs(
    csv_glob: str,
    parse_dates: Iterable[str],
    max_rows_total: int | None = None,
) -> pd.DataFrame:
    """Load and concatenate CSVs matching the glob, with optional truncation.

    Handles missing date columns by intersecting requested parse_dates with the
    actual columns present in each file, so missing optional columns do not fail
    the load.
    """

    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched glob: {csv_glob}")

    frames: List[pd.DataFrame] = []
    for path in paths:
        logging.info("Reading %s", path)
        # Peek at columns to guard against missing optional parse_date fields.
        cols = pd.read_csv(path, nrows=0).columns
        to_parse = [col for col in parse_dates if col in cols]
        missing = [col for col in parse_dates if col not in cols]
        if missing:
            logging.warning("Skipping parse_dates %s not present in %s", missing, path)
        frames.append(pd.read_csv(path, parse_dates=to_parse, low_memory=False))

    combined = pd.concat(frames, ignore_index=True)
    logging.info("Loaded %d rows from %d files", len(combined), len(paths))

    if max_rows_total is not None:
        combined = combined.iloc[:max_rows_total].copy()
        logging.info("Truncated to %d rows due to test mode cap", len(combined))

    return combined


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the DataFrame contains the required columns."""

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a DataFrame to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("Saved %d rows to %s", len(df), path)


def add_utm_coordinates(df: pd.DataFrame, utm_crs: str = "epsg:32632") -> pd.DataFrame:
    """
    Convert longitude/latitude to UTM x/y coordinates and add columns 'x_utm', 'y_utm'.
    """

    if "longitude" not in df.columns or "latitude" not in df.columns:
        raise ValueError("Longitude and latitude columns are required for UTM conversion.")

    transformer = Transformer.from_crs("epsg:4326", utm_crs, always_xy=True)
    x_utm, y_utm = transformer.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    df = df.copy()
    df["x_utm"] = x_utm
    df["y_utm"] = y_utm
    logging.info("Added UTM coordinates using CRS=%s", utm_crs)
    return df
