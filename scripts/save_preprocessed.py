"""Save preprocessed flight data using the backbone pipeline preprocessing stage."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone_tracks.config import load_config
from backbone_tracks.io import add_utm_coordinates, ensure_required_columns, load_monthly_csvs, save_dataframe
from backbone_tracks.preprocessing import preprocess_flights
from backbone_tracks.segmentation import segment_flights


def configure_logging(cfg: dict, log_name_override: str | None = None) -> None:
    """Configure logging based on config settings."""

    logging_cfg = cfg.get("logging", {}) or {}
    level_name = str(logging_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(levelname)s | %(message)s"

    handlers = []
    log_dir = logging_cfg.get("dir")
    log_file = log_name_override or logging_cfg.get("filename")
    if log_dir and log_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / log_file, mode="a", encoding="utf-8"))
    handlers.append(logging.StreamHandler())

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def _next_preprocessed_id(preprocessed_dir: Path) -> int:
    existing = sorted(preprocessed_dir.glob("preprocessed_*.csv"))
    ids = []
    for path in existing:
        stem = path.stem.replace("preprocessed_", "")
        if stem.isdigit():
            ids.append(int(stem))
    return max(ids, default=0) + 1


def _append_registry(
    registry_path: Path,
    preprocessed_path: Path,
    cfg: dict,
    n_flights: int,
    n_rows: int,
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessing_cfg = cfg.get("preprocessing", {}) or {}
    smoothing_cfg = preprocessing_cfg.get("smoothing", {}) or {}
    resampling_cfg = preprocessing_cfg.get("resampling", {}) or {}
    interp_method = resampling_cfg.get("method", "time")
    n_points = resampling_cfg.get("n_points")
    smoothing = (
        smoothing_cfg.get("method", "none") if smoothing_cfg.get("enabled", False) else "none"
    )
    include_altitude = bool(cfg.get("features", {}).get("include_altitude", False))
    flow_keys = (cfg.get("flows", {}) or {}).get("flow_keys", [])

    header = (
        "| id | file | n_flights | n_rows | n_points | interpolation | smoothing | include_altitude | flow_keys |"
    )
    sep = "|---|---|---:|---:|---:|---|---|---|---|"
    line = (
        f"| {preprocessed_path.stem.replace('preprocessed_', '')} | {preprocessed_path} | "
        f"{n_flights} | {n_rows} | {n_points} | {interp_method} | {smoothing} | "
        f"{'yes' if include_altitude else 'no'} | {flow_keys} |"
    )

    if not registry_path.exists():
        registry_path.write_text(header + "\n" + sep + "\n" + line + "\n", encoding="utf-8")
        return

    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg, log_name_override="preprocess.log")

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
    filter_cfg = preprocessing_cfg.get("filter", {}) or {}
    serial_column = preprocessing_cfg.get("serial_column")
    serial_start = int(preprocessing_cfg.get("serial_start", 1))

    output_cfg = cfg.get("output", {}) or {}
    output_dir = Path(output_cfg.get("dir", "data")) / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_id = output_cfg.get("preprocessed_id")
    if preprocessed_id is None:
        preprocessed_id = _next_preprocessed_id(output_dir)
    out_path = output_dir / f"preprocessed_{int(preprocessed_id)}.csv"

    df = load_monthly_csvs(csv_glob=csv_glob, parse_dates=parse_dates, max_rows_total=max_rows)
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

    preprocessed = preprocess_flights(
        df,
        smoothing_cfg=smoothing_cfg,
        resampling_cfg=resampling_cfg,
        filter_cfg=filter_cfg,
        use_utm=use_utm,
        flow_keys=flow_keys,
    )

    if "flight_id" in preprocessed.columns:
        unique_ids = sorted(preprocessed["flight_id"].dropna().unique())
        mapping = {fid: idx + 1 for idx, fid in enumerate(unique_ids)}
        preprocessed["flight_id"] = preprocessed["flight_id"].map(mapping).astype(int)

    if serial_column:
        if "flight_id" not in preprocessed.columns:
            raise ValueError("Cannot assign serial numbers without a flight_id column.")
        unique_ids = sorted(preprocessed["flight_id"].dropna().unique())
        mapping = {fid: idx + serial_start for idx, fid in enumerate(unique_ids)}
        preprocessed[serial_column] = preprocessed["flight_id"].map(mapping).astype(int)

    save_dataframe(preprocessed, out_path)

    n_flights = int(preprocessed["flight_id"].nunique()) if "flight_id" in preprocessed.columns else 0
    n_rows = int(len(preprocessed))
    registry_path = Path("thesis") / "docs" / "preprocessed_registry.md"
    _append_registry(registry_path, out_path, cfg, n_flights, n_rows)
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
