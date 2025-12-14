"""Save preprocessed flight data using the backbone pipeline preprocessing stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone_tracks.config import load_config
from backbone_tracks.io import add_utm_coordinates, ensure_required_columns, load_monthly_csvs, save_dataframe
from backbone_tracks.preprocessing import preprocess_flights
from backbone_tracks.segmentation import segment_flights


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    input_cfg = cfg.get("input", {}) or {}
    csv_glob = input_cfg.get("csv_glob", "Enhanced/matched_*.csv")
    parse_dates = input_cfg.get("parse_dates", ["timestamp"])
    max_rows = None
    testing_cfg = cfg.get("testing", {}) or {}
    if testing_cfg.get("enabled", False):
        max_rows = testing_cfg.get("max_rows_total")

    coord_cfg = cfg.get("coordinates", {}) or {}
    use_utm = bool(coord_cfg.get("use_utm", False))
    utm_crs = coord_cfg.get("utm_crs", "epsg:32632")

    flows_cfg = cfg.get("flows", {}) or {}
    flow_keys = flows_cfg.get("flow_keys", ["Runway"])

    seg_cfg = cfg.get("segmentation", {}) or {}
    preprocessing_cfg = cfg.get("preprocessing", {}) or {}
    smoothing_cfg = preprocessing_cfg.get("smoothing", {})
    resampling_cfg = preprocessing_cfg.get("resampling", {})

    output_cfg = cfg.get("output", {}) or {}
    exp_name = str(output_cfg.get("experiment_name", "preprocessed"))
    output_dir = Path(output_cfg.get("dir", "output")) / "run" / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"preprocessed_{exp_name}.csv"

    df = load_monthly_csvs(csv_glob=csv_glob, parse_dates=parse_dates, max_rows_total=max_rows)
    df = ensure_required_columns(df)
    if use_utm:
        df = add_utm_coordinates(df, utm_crs=utm_crs)

    df = segment_flights(
        df,
        time_gap_sec=float(seg_cfg.get("time_gap_sec", 60)),
        distance_jump_m=float(seg_cfg.get("distance_jump_m", 600)),
        min_points_per_flight=int(seg_cfg.get("min_points_per_flight", 10)),
    )

    preprocessed = preprocess_flights(
        df,
        smoothing_cfg=smoothing_cfg,
        resampling_cfg=resampling_cfg,
        use_utm=use_utm,
        flow_keys=flow_keys,
    )

    save_dataframe(preprocessed, out_path)
    print(f"Saved preprocessed data to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save preprocessed flight data.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/backbone_full.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)
