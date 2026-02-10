"""Pre-assess raw flight trajectory lengths before preprocessing.

Usage:
  python scripts/eda_raw_lengths.py -c config/backbone_full.yaml --outdir output/eda/raw_lengths

Outputs:
  - flight_lengths.csv
  - length_summary.json
  - length_histogram.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone_tracks.config import load_config
from backbone_tracks.io import add_utm_coordinates, ensure_required_columns, load_monthly_csvs
from backbone_tracks.segmentation import segment_flights


def _compute_lengths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["flight_id", "timestamp"]).copy()
    dx = df.groupby("flight_id")["x_utm"].diff()
    dy = df.groupby("flight_id")["y_utm"].diff()
    seg = np.sqrt(dx * dx + dy * dy)
    lengths = seg.groupby(df["flight_id"]).sum().fillna(0.0)
    meta_cols = ["A/D", "Runway", "icao24"]
    meta = df.groupby("flight_id").first()[[c for c in meta_cols if c in df.columns]]
    out = pd.DataFrame({"flight_id": lengths.index, "length_m": lengths.values}).set_index("flight_id")
    out = out.join(meta)
    return out.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute raw flight length statistics.")
    parser.add_argument("-c", "--config", default="config/backbone_full.yaml")
    parser.add_argument("--outdir", default="output/eda/raw_lengths")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_cfg = cfg.get("input", {}) or {}
    csv_glob = input_cfg.get("csv_glob", "data/merged/matched_trajs_*.csv")
    parse_dates = input_cfg.get("parse_dates", ["timestamp"])

    coord_cfg = cfg.get("coordinates", {}) or {}
    use_utm = bool(coord_cfg.get("use_utm", True))
    utm_crs = coord_cfg.get("utm_crs", "epsg:32632")

    seg_cfg = cfg.get("segmentation", {}) or {}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_monthly_csvs(csv_glob=csv_glob, parse_dates=parse_dates, max_rows_total=None)
    df = ensure_required_columns(df)
    if use_utm:
        df = add_utm_coordinates(df, utm_crs=utm_crs)

    df = segment_flights(
        df,
        time_gap_sec=float(seg_cfg.get("time_gap_sec", 60)),
        distance_jump_m=float(seg_cfg.get("distance_jump_m", 600)),
        min_points_per_flight=int(seg_cfg.get("min_points_per_flight", 10)),
        split_on_identity=bool(seg_cfg.get("split_on_identity", True)),
    )

    lengths = _compute_lengths(df)
    lengths.to_csv(outdir / "flight_lengths.csv", index=False)

    summary = {
        "n_flights": int(len(lengths)),
        "length_median_m": float(lengths["length_m"].median()),
        "length_p10_m": float(lengths["length_m"].quantile(0.10)),
        "length_p25_m": float(lengths["length_m"].quantile(0.25)),
        "length_p75_m": float(lengths["length_m"].quantile(0.75)),
        "length_p90_m": float(lengths["length_m"].quantile(0.90)),
    }
    (outdir / "length_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 4))
    plt.hist(lengths["length_m"].values, bins=60, color="#4E79A7", alpha=0.85)
    plt.title("Flight length distribution (raw, segmented)")
    plt.xlabel("length (m)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outdir / "length_histogram.png", dpi=150)
    plt.close()

    print(f"Wrote {outdir / 'flight_lengths.csv'}")
    print(f"Wrote {outdir / 'length_summary.json'}")


if __name__ == "__main__":
    main()
