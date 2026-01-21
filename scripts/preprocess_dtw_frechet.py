"""
Preprocess a trajectory CSV for DTW/Frechet experiments by filtering short flights
and resampling each flight to a fixed number of points.

Usage:
  python scripts/preprocess_dtw_frechet.py ^
    --input output/run/csv/preprocessed_OPTICS_exp_5.csv ^
    --output output/run/csv/preprocessed_OPTICS_exp_5_lenmed_top80_n40.csv ^
    --target-points 40 ^
    --keep-top-pct 0.8 ^
    --min-length-mode median

Input format (columns expected in the CSV):
  step, x_utm, y_utm, altitude, dist_to_airport_m, latitude, longitude,
  A/D, Runway, flight_id, icao24, callsign, aircraft_type_match

Output format (same columns as input, with step reset 0..target-1):
  step, x_utm, y_utm, altitude, dist_to_airport_m, latitude, longitude,
  A/D, Runway, flight_id, icao24, callsign, aircraft_type_match
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

INPUT_COLUMNS = [
    "step",
    "x_utm",
    "y_utm",
    "altitude",
    "dist_to_airport_m",
    "latitude",
    "longitude",
    "A/D",
    "Runway",
    "flight_id",
    "icao24",
    "callsign",
    "aircraft_type_match",
]

INTERP_COLUMNS = [
    "x_utm",
    "y_utm",
    "altitude",
    "dist_to_airport_m",
    "latitude",
    "longitude",
]

CONST_COLUMNS = [
    "A/D",
    "Runway",
    "flight_id",
    "icao24",
    "callsign",
    "aircraft_type_match",
]


@dataclass
class FilterConfig:
    """Filter settings for trajectory selection."""

    target_points: int
    keep_top_pct: float
    min_length_mode: str


def compute_flight_metrics(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """Compute length, displacement, and point count per flight_id.

    Returns a DataFrame indexed by flight_id with columns:
      n_points, length_m, displacement_m
    """

    dx = df_sorted.groupby("flight_id")["x_utm"].diff()
    dy = df_sorted.groupby("flight_id")["y_utm"].diff()
    seg_dist = np.sqrt(dx * dx + dy * dy)

    lengths = seg_dist.groupby(df_sorted["flight_id"]).sum().fillna(0.0)
    n_points = df_sorted.groupby("flight_id").size()

    first = df_sorted.groupby("flight_id")[["x_utm", "y_utm"]].first()
    last = df_sorted.groupby("flight_id")[["x_utm", "y_utm"]].last()
    displacement = np.sqrt(((last - first) ** 2).sum(axis=1))

    metrics = pd.DataFrame(
        {
            "n_points": n_points,
            "length_m": lengths,
            "displacement_m": displacement,
        }
    )
    return metrics


def determine_length_threshold(metrics: pd.DataFrame, config: FilterConfig) -> Dict[str, float]:
    """Determine length thresholds based on median and top-pct criteria."""

    median_len = float(metrics["length_m"].median())
    p20_len = float(metrics["length_m"].quantile(1.0 - config.keep_top_pct))
    threshold_len = max(median_len, p20_len) if config.min_length_mode == "median" else p20_len

    return {
        "median_len": median_len,
        "p20_len": p20_len,
        "threshold_len": threshold_len,
    }


def resample_flight(df_flight: pd.DataFrame, target_points: int) -> Tuple[pd.DataFrame | None, float]:
    """Resample a single flight to target_points along cumulative distance.

    Returns (resampled_df, length_m_resampled). If length is zero, returns (None, 0).
    """

    df_flight = df_flight.sort_values("step")
    xy = df_flight[["x_utm", "y_utm"]].to_numpy()
    if len(xy) < 2:
        return None, 0.0

    seg = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if total <= 0.0:
        return None, 0.0

    target = np.linspace(0.0, total, target_points)
    out = {"step": np.arange(target_points, dtype=int)}
    for col in INTERP_COLUMNS:
        out[col] = np.interp(target, cum, df_flight[col].to_numpy())

    first_row = df_flight.iloc[0]
    for col in CONST_COLUMNS:
        out[col] = first_row[col]

    resampled = pd.DataFrame(out, columns=INPUT_COLUMNS)

    # Compute length of resampled path for validation.
    xy_new = resampled[["x_utm", "y_utm"]].to_numpy()
    seg_new = np.sqrt(np.sum(np.diff(xy_new, axis=0) ** 2, axis=1))
    length_new = float(np.sum(seg_new))
    return resampled, length_new


def write_metadata(
    out_meta: Path,
    input_path: Path,
    output_path: Path,
    config: FilterConfig,
    thresholds: Dict[str, float],
    kept_count: int,
    total_count: int,
    length_error_stats: Dict[str, float],
) -> None:
    """Write a JSON metadata summary alongside the output CSV."""

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "target_points": config.target_points,
        "keep_top_pct": config.keep_top_pct,
        "min_length_mode": config.min_length_mode,
        "thresholds_m": thresholds,
        "flights_total": total_count,
        "flights_kept": kept_count,
        "flights_removed": total_count - kept_count,
        "resampled_length_error": length_error_stats,
        "notes": "Filtered by length threshold and resampled by cumulative distance.",
    }
    out_meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter short flights and resample trajectories for DTW/Frechet."
    )
    parser.add_argument("--input", required=True, help="Input preprocessed CSV.")
    parser.add_argument("--output", required=True, help="Output CSV (resampled).")
    parser.add_argument("--target-points", type=int, default=40)
    parser.add_argument("--keep-top-pct", type=float, default=0.8)
    parser.add_argument("--min-length-mode", choices=["median", "top_pct"], default="median")
    args = parser.parse_args()

    config = FilterConfig(
        target_points=args.target_points,
        keep_top_pct=args.keep_top_pct,
        min_length_mode=args.min_length_mode,
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_path.with_suffix(".meta.json")
    removed_path = output_path.with_suffix(".removed.csv")

    # Ensure a clean write on reruns.
    if output_path.exists():
        output_path.unlink()
    if meta_path.exists():
        meta_path.unlink()
    if removed_path.exists():
        removed_path.unlink()

    df = pd.read_csv(input_path)
    df = df[INPUT_COLUMNS]
    df_sorted = df.sort_values(["flight_id", "step"]).reset_index(drop=True)

    metrics = compute_flight_metrics(df_sorted)
    thresholds = determine_length_threshold(metrics, config)
    keep_mask = metrics["length_m"] >= thresholds["threshold_len"]
    keep_ids = set(metrics.index[keep_mask])

    # Prepare removal report.
    removed = metrics[~keep_mask].copy()
    removed["reason"] = "length_below_threshold"
    removed.to_csv(removed_path, index_label="flight_id")

    # Stream output to avoid holding all resampled data in memory.
    wrote_header = False
    length_ratios: List[float] = []
    kept_count = 0

    for flight_id, group in df_sorted.groupby("flight_id", sort=False):
        if flight_id not in keep_ids:
            continue
        resampled, length_new = resample_flight(group, config.target_points)
        if resampled is None:
            continue
        length_orig = float(metrics.loc[flight_id, "length_m"])
        if length_orig > 0.0:
            length_ratios.append(length_new / length_orig)
        resampled.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        kept_count += 1

    total_count = int(metrics.shape[0])
    length_error_stats = {
        "ratio_mean": float(np.mean(length_ratios)) if length_ratios else 0.0,
        "ratio_median": float(np.median(length_ratios)) if length_ratios else 0.0,
        "ratio_min": float(np.min(length_ratios)) if length_ratios else 0.0,
        "ratio_max": float(np.max(length_ratios)) if length_ratios else 0.0,
    }

    write_metadata(
        meta_path,
        input_path,
        output_path,
        config,
        thresholds,
        kept_count,
        total_count,
        length_error_stats,
    )

    print(f"Wrote {kept_count} flights to {output_path}")
    print(f"Removed flights report: {removed_path}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
