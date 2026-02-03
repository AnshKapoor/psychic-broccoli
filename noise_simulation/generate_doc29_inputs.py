"""Generate Doc29 groundtracks + Flight CSVs from exported 7-track layouts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from .ground_track_generation_for_ansh import groundtrack_interpolation
except ImportError:  # pragma: no cover - fallback for script execution
    from ground_track_generation_for_ansh import groundtrack_interpolation

TRACK_DEFS = [
    ("center", 0.28),
    ("inner_left", 0.22),
    ("inner_right", 0.22),
    ("middle_left", 0.11),
    ("middle_right", 0.11),
    ("outer_left", 0.03),
    ("outer_right", 0.03),
]

MODE_MAP = {
    "Start": "take-off",
    "Landung": "landing",
}

RUNWAY_FILE_MAP = {
    "09R": "Runways/09R-27L-HAJ.xml",
    "27L": "Runways/09R-27L-HAJ.xml",
    "09L": "Runways/09L-27R-HAJ.xml",
    "27R": "Runways/09L-27R-HAJ.xml",
}

STARTPOINTS = {
    ("27L", "landing"): (546026.9, 5811859.4),
    ("27L", "take-off"): (548229.4, 5811780.4),
    ("09R", "landing"): (548229.4, 5811780.4),
    ("09R", "take-off"): (546026.9, 5811859.4),
    ("27R", "landing"): (544424.6, 5813317.4),
    ("27R", "take-off"): (547474.3, 5813208.8),
    ("09L", "landing"): (547474.3, 5813208.8),
    ("09L", "take-off"): (544424.6, 5813317.4),
}


def _parse_flow_dir(flow_dir: Path) -> Tuple[str, str]:
    parts = flow_dir.name.split("_", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected flow folder name: {flow_dir.name}")
    return parts[0], parts[1]


def _resolve_track_dir(flow_dir: Path, cluster_id: int | None) -> Path:
    """Return the directory that contains groundtrack_*.csv files."""

    if (flow_dir / "groundtrack_center.csv").exists():
        return flow_dir

    cluster_dirs = [p for p in flow_dir.iterdir() if p.is_dir() and p.name.startswith("cluster_")]
    if cluster_id is not None:
        target = flow_dir / f"cluster_{cluster_id}"
        if not target.exists():
            raise FileNotFoundError(f"Missing cluster directory: {target}")
        return target

    if len(cluster_dirs) == 1:
        return cluster_dirs[0]

    raise ValueError(
        f"Ambiguous track directories under {flow_dir}. "
        "Specify --cluster-id to select a cluster subdirectory."
    )


def _format_first_point(runway: str, mode: str) -> str:
    if (runway, mode) not in STARTPOINTS:
        raise ValueError(f"Unsupported runway/mode: {runway} {mode}")
    x, y = STARTPOINTS[(runway, mode)]
    return f"{x}, {y}"


def _generate_discrete_groundtrack(
    raw_path: Path,
    out_path: Path,
    runway: str,
    mode: str,
    interpolation_length: float,
) -> None:
    df = pd.read_csv(raw_path, sep=";")
    if "easting" not in df.columns or "northing" not in df.columns:
        raise ValueError(f"Missing easting/northing in {raw_path}")

    # Mirror the original script: drop the first row, then insert runway start point.
    if len(df) > 0:
        df = df.drop(index=0).reset_index(drop=True)

    if (runway, mode) not in STARTPOINTS:
        raise ValueError(f"Unsupported runway/mode: {runway} {mode}")
    start_x, start_y = STARTPOINTS[(runway, mode)]
    start_row = pd.DataFrame({"easting": [start_x], "northing": [start_y]})
    df = pd.concat([start_row, df[["easting", "northing"]]], ignore_index=True)

    interpolator = groundtrack_interpolation()
    interp = interpolator.interpolation(df, interpolation_length)
    interp = interpolator.calculate_turn_radius(interp)
    interp.rename(columns={"turn_radius": "radius"}, inplace=True)

    distances = (interp["easting"].diff() ** 2 + interp["northing"].diff() ** 2) ** 0.5
    interp["s"] = distances.cumsum().fillna(0)
    interp.to_csv(out_path, sep=";", index=True)


def _compute_track_counts(total: int) -> Dict[str, int]:
    counts = {}
    total_weighted = 0
    for name, weight in TRACK_DEFS:
        count = int(round(total * weight))
        counts[name] = count
        total_weighted += count
    diff = total - total_weighted
    counts["center"] = max(counts["center"] + diff, 0)
    return counts


def _build_flight_csv(
    columns: Dict[str, List[str]],
    out_path: Path,
) -> None:
    row_labels = [
        "mode",
        "default startpoint",
        "first point",
        "reference speed",
        "engine type",
        "engine position",
        "NPD table",
        "Groundtrack",
        "Runway",
        "Height Profile",
        "Nr. day",
        "Nr. night",
        "sc_data",
    ]
    data = {"Unnamed: 0": row_labels}
    data.update(columns)
    df = pd.DataFrame(data)
    df.to_csv(out_path, sep=";", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Doc29 groundtracks + Flight CSVs for EXP runs.")
    parser.add_argument("--tracks-root", required=True, help="Root folder with doc29_tracks/<A_D>_<Runway>/groundtrack_*.csv files.")
    parser.add_argument("--preprocessed", required=True, help="Preprocessed CSV to count flights per flow.")
    parser.add_argument("--experiment-name", required=True, help="Experiment code (e.g., EXP46).")
    parser.add_argument("--output-root", default="noise_simulation/doc-29-implementation", help="Doc29 implementation root.")
    parser.add_argument("--interpolation-length", type=float, default=200.0, help="Interpolation spacing in metres.")
    parser.add_argument("--npd-arrival", default="NPD_data/CFM563_A.csv", help="NPD table for landings.")
    parser.add_argument("--npd-departure", default="NPD_data/CFM563_D.csv", help="NPD table for take-offs.")
    parser.add_argument("--height-arrival", default="Flight_profiles/reference_arrival.csv", help="Height profile for landings.")
    parser.add_argument("--height-departure", default="Flight_profiles/reference_departure.csv", help="Height profile for take-offs.")
    parser.add_argument("--spectral-class", default="Atmosphere_model/Spectral_classes.csv", help="Spectral class file.")
    parser.add_argument("--reference-speed", default="160", help="Reference speed.")
    parser.add_argument("--engine-type", default="turbofan", help="Engine type.")
    parser.add_argument("--engine-position", default="wing-mounted", help="Engine position.")
    parser.add_argument("--default-startpoint", default="True", help="Default startpoint (True/False).")
    parser.add_argument("--nr-night", default="0", help="Number of night flights per track.")
    parser.add_argument("--combine", action="store_true", help="Write a combined Flight_<EXP>.csv across all flows.")
    parser.add_argument("--cluster-id", type=int, default=None, help="Select cluster subdirectory if present (e.g., cluster_0).")
    parser.add_argument(
        "--skip-unknown-runways",
        action="store_true",
        help="Skip flows with runways not present in RUNWAY_FILE_MAP (e.g., 09C/27C/HEL).",
    )
    args = parser.parse_args()

    tracks_root = Path(args.tracks_root)
    preprocessed = Path(args.preprocessed)
    output_root = Path(args.output_root)
    groundtracks_root = output_root / "Groundtracks" / args.experiment_name
    groundtracks_root.mkdir(parents=True, exist_ok=True)

    df_pre = pd.read_csv(preprocessed)
    if not {"A/D", "Runway", "flight_id"}.issubset(df_pre.columns):
        raise ValueError("Preprocessed CSV must include A/D, Runway, and flight_id columns.")
    flight_counts = (
        df_pre[["A/D", "Runway", "flight_id"]]
        .dropna()
        .drop_duplicates()
        .groupby(["A/D", "Runway"])
        .size()
    )

    combined_columns: Dict[str, List[str]] = {}

    for flow_dir in sorted(tracks_root.iterdir()):
        if not flow_dir.is_dir():
            continue
        ad, runway = _parse_flow_dir(flow_dir)
        if ad not in MODE_MAP:
            raise ValueError(f"Unsupported A/D label: {ad}")
        mode = MODE_MAP[ad]
        if runway not in RUNWAY_FILE_MAP:
            if args.skip_unknown_runways:
                print(f"[skip] Unsupported runway {runway} for flow {ad}/{runway}")
                continue
            raise ValueError(f"Unsupported runway: {runway}")

        track_dir = _resolve_track_dir(flow_dir, args.cluster_id)
        total_flights = int(flight_counts.get((ad, runway), 0))
        if total_flights <= 0:
            raise ValueError(f"No flights found for flow {ad}/{runway} in {preprocessed}")

        track_counts = _compute_track_counts(total_flights)
        flow_out_dir = groundtracks_root / f"{ad}_{runway}"
        flow_out_dir.mkdir(parents=True, exist_ok=True)

        columns: Dict[str, List[str]] = {}
        for track_name, _weight in TRACK_DEFS:
            raw_track = track_dir / f"groundtrack_{track_name}.csv"
            if not raw_track.exists():
                raise FileNotFoundError(f"Missing track file: {raw_track}")
            out_name = f"{args.experiment_name}_{ad}_{runway}_{track_name}.csv"
            out_path = flow_out_dir / out_name

            _generate_discrete_groundtrack(
                raw_track,
                out_path,
                runway=runway,
                mode=mode,
                interpolation_length=args.interpolation_length,
            )

            col_name = f"{args.experiment_name}_{ad}_{runway}_{track_name}"
            npd_table = args.npd_arrival if mode == "landing" else args.npd_departure
            height_profile = args.height_arrival if mode == "landing" else args.height_departure
            columns[col_name] = [
                mode,
                args.default_startpoint,
                _format_first_point(runway, mode),
                args.reference_speed,
                args.engine_type,
                args.engine_position,
                npd_table,
                f"Groundtracks/{args.experiment_name}/{ad}_{runway}/{out_name}",
                RUNWAY_FILE_MAP[runway],
                height_profile,
                str(track_counts[track_name]),
                args.nr_night,
                args.spectral_class,
            ]

        flow_flight_path = output_root / f"Flight_{args.experiment_name}_{ad}_{runway}.csv"
        _build_flight_csv(columns, flow_flight_path)

        if args.combine:
            combined_columns.update(columns)

    if args.combine and combined_columns:
        combined_path = output_root / f"Flight_{args.experiment_name}.csv"
        _build_flight_csv(combined_columns, combined_path)

    print(f"Doc29 inputs written under {groundtracks_root}")


if __name__ == "__main__":
    main()
