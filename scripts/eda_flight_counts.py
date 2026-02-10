"""Compute flight counts by runway, operation type, and aircraft type.

Usage:
  python scripts/eda_flight_counts.py -c config/backbone_full.yaml --outdir output/eda/flight_counts
  python scripts/eda_flight_counts.py --csv-glob matched_trajectories/matched_trajs_*.csv --outdir output/eda/flight_counts_matched

Notes:
  - By default uses the config's input csv_glob.
  - Use --csv-glob to override the input files (e.g., matched_trajectories).

Outputs:
  - flight_counts_overall.json
  - flights_by_runway.csv
  - flights_by_operation.csv
  - flights_by_aircraft_type.csv
  - flights_by_runway_operation.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone_tracks.config import load_config
from backbone_tracks.io import add_utm_coordinates, ensure_required_columns, load_monthly_csvs
from backbone_tracks.segmentation import segment_flights


def _pick_aircraft_type_column(df: pd.DataFrame) -> str | None:
    candidates = ["aircraft_type", "typecode", "icao_type", "aircraft"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _lookup_aircraft_types(icao24_values: pd.Series, cache_path: Path) -> dict[str, str]:
    """Resolve aircraft type codes via the traffic library and cache to CSV."""
    icao24_values = icao24_values.dropna().astype(str).str.lower().unique().tolist()
    if not icao24_values:
        return {}

    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if {"icao24", "aircraft_type"}.issubset(cached.columns):
            cached["icao24"] = cached["icao24"].astype(str).str.lower()
            cached["aircraft_type"] = cached["aircraft_type"].astype(str)
            return dict(zip(cached["icao24"], cached["aircraft_type"]))

    try:
        from traffic.data import aircraft as aircraft_db
    except Exception:
        return {}

    rows: list[dict[str, str]] = []
    for icao in icao24_values:
        ac = aircraft_db.get(icao)
        if ac is None:
            continue
        typecode = getattr(ac, "typecode", None) or getattr(ac, "model", None)
        if typecode:
            rows.append({"icao24": icao, "aircraft_type": str(typecode).strip().upper()})

    if rows:
        df_cache = pd.DataFrame(rows).drop_duplicates("icao24")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_cache.to_csv(cache_path, index=False)
        return dict(zip(df_cache["icao24"], df_cache["aircraft_type"]))
    return {}


def _write_counts(df: pd.DataFrame, outdir: Path) -> None:
    total_flights = int(df["flight_id"].nunique())
    total_rows = int(len(df))

    summary = {
        "total_flights": total_flights,
        "total_rows": total_rows,
    }
    (outdir / "flight_counts_overall.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "Runway" in df.columns:
        by_runway = df.groupby("Runway")["flight_id"].nunique().reset_index(name="n_flights")
        by_runway.to_csv(outdir / "flights_by_runway.csv", index=False)

    if "A/D" in df.columns:
        by_op = df.groupby("A/D")["flight_id"].nunique().reset_index(name="n_flights")
        by_op.to_csv(outdir / "flights_by_operation.csv", index=False)

    if "Runway" in df.columns and "A/D" in df.columns:
        by_runway_op = (
            df.groupby(["A/D", "Runway"])["flight_id"].nunique().reset_index(name="n_flights")
        )
        by_runway_op.to_csv(outdir / "flights_by_runway_operation.csv", index=False)

    ac_col = _pick_aircraft_type_column(df)
    if ac_col:
        by_type = df.groupby(ac_col)["flight_id"].nunique().reset_index(name="n_flights")
        by_type = by_type.sort_values("n_flights", ascending=False)
        by_type.to_csv(outdir / "flights_by_aircraft_type.csv", index=False)
    elif "icao24" in df.columns:
        cache_path = outdir / "aircraft_type_by_icao24.csv"
        type_map = _lookup_aircraft_types(df["icao24"], cache_path)
        if type_map:
            df = df.copy()
            df["aircraft_type"] = df["icao24"].astype(str).str.lower().map(type_map)
            by_type = (
                df.groupby("aircraft_type")["flight_id"].nunique().reset_index(name="n_flights")
            )
            by_type = by_type.sort_values("n_flights", ascending=False)
            by_type.to_csv(outdir / "flights_by_aircraft_type.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute flight counts by grouping columns.")
    parser.add_argument("-c", "--config", default="config/backbone_full.yaml")
    parser.add_argument("--outdir", default="output/eda/flight_counts")
    parser.add_argument(
        "--csv-glob",
        default=None,
        help="Override input CSV glob (e.g., matched_trajectories/matched_trajs_*.csv).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_cfg = cfg.get("input", {}) or {}
    csv_glob = args.csv_glob or input_cfg.get("csv_glob", "data/merged/matched_trajs_*.csv")
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

    _write_counts(df, outdir)
    print(f"Wrote flight count summaries to {outdir}")


if __name__ == "__main__":
    main()
