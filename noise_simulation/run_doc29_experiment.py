"""Automate Doc29 simulations per experiment, cluster, and aircraft type.

Usage:
  python noise_simulation/run_doc29_experiment.py --experiment EXP46_optics_euclid_ms8_xi03_mcs06
  python noise_simulation/run_doc29_experiment.py --experiment EXP46_optics_euclid_ms8_xi03_mcs06 --aircraft-types A320 A321

Example input formats:
  - labels csv: columns flight_id, cluster_id, A/D, Runway
  - preprocessed CSV: columns step, x_utm, y_utm, flight_id, A/D, Runway, icao24
  - matched CSV: columns icao24, aircraft_type_adsb, aircraft_type_noise

Example output tree:
  noise_simulation/results/<EXP>/<A_D>_<Runway>/cluster_<id>/<type>/
    Flight_subtracks.csv
    Flight_groundtruth.csv
    subtracks.csv
    groundtruth.csv
    mse.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noise_simulation import generate_doc29_inputs as doc29_inputs
from noise_simulation.automation.aircraft_types import (
    build_flight_meta,
    build_flight_type_map,
    build_icao_type_map,
    compute_aircraft_type_counts,
    load_or_build_top10,
    NPD_TYPE_MAP,
    NPD_ID_TO_ICAO,
    FLIGHT_PROFILE_MAP,
    SPECTRAL_CLASS_MAP,
)
from noise_simulation.automation.compare import compare_cumulative_res
from noise_simulation.automation.doc29_runner import run_doc29
from noise_simulation.automation.experiment_paths import (
    load_experiment_config,
    list_label_files,
    parse_flow_from_label,
)
from noise_simulation.automation.flight_csv import (
    FlightCsvConfig,
    FlightEntry,
    build_columns,
    write_flight_csv,
)
from noise_simulation.automation.groundtracks import ensure_raw_tracks, generate_discrete_tracks
from noise_simulation.automation.groundtruth_tracks import generate_groundtruth_tracks


def _npd_suffix(ad: str) -> str:
    """Return NPD suffix for A/D.

    Usage:
      suffix = _npd_suffix("Landung")  # "A"
    """
    if ad == "Landung":
        return "A"
    if ad == "Start":
        return "D"
    raise ValueError(f"Unsupported A/D label: {ad}")


def _safe_name(value: str) -> str:
    """Return a filesystem-friendly name.

    Usage:
      safe = _safe_name("B738/CFM")
    """
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _type_folder_name(icao_type: str, npd_id: str) -> str:
    """Return a collision-safe folder name for type outputs.

    Usage:
      folder = _type_folder_name("E75L", "CF348E")  # E75L__CF348E
    """
    return _safe_name(f"{icao_type}__{npd_id}")


def _relative_to_doc29(doc29_root: Path, path: Path) -> str:
    """Return POSIX-style path relative to doc29_root.

    Usage:
      rel = _relative_to_doc29(doc29_root, doc29_root / "NPD_data" / "B734_A.csv")
    """
    return path.relative_to(doc29_root).as_posix()


def _warn_missing_profiles(logger: logging.Logger, doc29_root: Path) -> None:
    profiles_dir = doc29_root / "Flight_profiles" / "Profiles for Ansh"
    for icao, acft_id in FLIGHT_PROFILE_MAP.items():
        for suffix in ("A", "D"):
            path = profiles_dir / f"{acft_id}_{suffix}.csv"
            if not path.exists():
                logger.warning("Missing height profile for %s: %s", icao, path)


def _resolve_height_profile(doc29_root: Path, aircraft_type: str, ad: str) -> str:
    """Resolve aircraft-specific height profile, fallback to reference.

    Expects profiles at:
      Flight_profiles/Profiles for Ansh/{ACFT_ID}_{A|D}.csv
    """
    suffix = "A" if ad == "Landung" else "D"
    acft_id = FLIGHT_PROFILE_MAP.get(aircraft_type)
    if acft_id:
        candidate = doc29_root / "Flight_profiles" / "Profiles for Ansh" / f"{acft_id}_{suffix}.csv"
        if candidate.exists():
            return _relative_to_doc29(doc29_root, candidate)
    # fallback
    ref = doc29_root / "Flight_profiles" / ("reference_arrival.csv" if ad == "Landung" else "reference_departure.csv")
    return _relative_to_doc29(doc29_root, ref)


def _resolve_spectral_class(doc29_root: Path, aircraft_type: str, ad: str) -> str:
    """Resolve spectral class file by aircraft + operation, fallback to default."""
    key = "approach" if ad == "Landung" else "departure"
    class_id = SPECTRAL_CLASS_MAP.get(aircraft_type, {}).get(key)
    if class_id is not None:
        candidate = doc29_root / "Atmosphere_model" / f"Spectral_class_{class_id}.csv"
        if candidate.exists():
            return _relative_to_doc29(doc29_root, candidate)
    return _relative_to_doc29(doc29_root, doc29_root / "Atmosphere_model" / "Spectral_classes.csv")


def _build_flight_cfg(
    ad: str,
    runway: str,
    npd_table: str,
    height_profile: str,
    spectral_class: str,
    reference_speed: str,
    engine_type: str,
    engine_position: str,
    default_startpoint: str,
) -> FlightCsvConfig:
    """Build shared flight.csv config for a single A/D + runway.

    Usage:
      cfg = _build_flight_cfg("Start", "09L", "NPD_data/B738_D.csv", ...)
    """
    mode = doc29_inputs.MODE_MAP[ad]
    return FlightCsvConfig(
        mode=mode,
        default_startpoint=default_startpoint,
        first_point=doc29_inputs._format_first_point(runway, mode),
        reference_speed=reference_speed,
        engine_type=engine_type,
        engine_position=engine_position,
        npd_table=npd_table,
        runway_file=doc29_inputs.RUNWAY_FILE_MAP[runway],
        height_profile=height_profile,
        nr_night="0",
        spectral_class=spectral_class,
    )


def _configure_logging(log_file: Path, experiment: str) -> logging.Logger:
    """Configure a file + console logger with timestamps and experiment name.

    Usage:
      logger = _configure_logging(Path("run.log"), "EXP46_optics_euclid_ms8_xi03_mcs06")
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("doc29_automation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = "%(asctime)s | %(levelname)s | %(filename)s | exp=%(experiment)s | %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    class _ExperimentAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            extra = kwargs.get("extra", {})
            extra.setdefault("experiment", experiment)
            kwargs["extra"] = extra
            return msg, kwargs

    return _ExperimentAdapter(logger, {"experiment": experiment})


def main() -> None:
    parser = argparse.ArgumentParser(description="Automate Doc29 simulations for an experiment.")
    parser.add_argument("--experiment", required=True, help="Experiment folder name (e.g., EXP46_optics_euclid_ms8_xi03_mcs06).")
    parser.add_argument("--doc29-root", default="noise_simulation/doc-29-implementation", help="Doc29 implementation root.")
    parser.add_argument("--results-root", default=None, help="Results root (defaults to noise_simulation/results/<EXP>).")
    parser.add_argument("--top10-cache", default="output/eda/aircraft_type_counts.csv", help="Cache for top-10 aircraft types.")
    parser.add_argument("--matched-dir", default="matched_trajectories", help="Directory containing matched_trajs_*.csv.")
    parser.add_argument("--interpolation-length", type=float, default=200.0, help="Groundtrack interpolation spacing in metres.")
    parser.add_argument("--cumulative-metric", default="Leq_day", help="Doc29 cumulative metric.")
    parser.add_argument("--time-in-s", type=float, default=86400 * 365 * (2 / 3), help="Time horizon in seconds.")
    parser.add_argument("--reference-speed", default="160", help="Reference speed for flight.csv.")
    parser.add_argument("--engine-type", default="turbofan", help="Engine type for flight.csv.")
    parser.add_argument("--engine-position", default="wing-mounted", help="Engine position for flight.csv.")
    parser.add_argument("--default-startpoint", default="True", help="Default startpoint (True/False).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip simulations if outputs already exist.")
    parser.add_argument(
        "--aircraft-types",
        nargs="+",
        default=None,
        help="Restrict to specific aircraft types (e.g., A320 A321).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path (defaults to noise_simulation/results/<EXP>/run.log).",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    exp_dir = repo_root / "output" / "experiments" / args.experiment
    cfg, preprocessed_path = load_experiment_config(repo_root, args.experiment)
    label_files = list_label_files(exp_dir)

    matched_dir = repo_root / args.matched_dir
    matched_paths = sorted(matched_dir.glob("matched_trajs_*.csv"))
    if not matched_paths:
        raise FileNotFoundError(f"No matched_trajs_*.csv in {matched_dir}")

    flight_meta = build_flight_meta(preprocessed_path)
    icao_set = {meta["icao24"] for meta in flight_meta.values() if meta["icao24"] != "UNKNOWN"}
    icao_map = build_icao_type_map(matched_paths, icao_set)
    flight_type_map = build_flight_type_map(flight_meta, icao_map)
    counts_df = compute_aircraft_type_counts(flight_type_map)
    if args.aircraft_types:
        top10 = []
        for t in args.aircraft_types:
            if not t or not str(t).strip():
                continue
            raw = str(t).strip().upper()
            # Accept ICAO or NPD IDs; normalize to ICAO for profile/spectral mapping.
            if raw in NPD_ID_TO_ICAO:
                top10.append(NPD_ID_TO_ICAO[raw])
            else:
                top10.append(raw)
    else:
        top10 = [t for t in load_or_build_top10(counts_df, repo_root / args.top10_cache) if t != "UNKNOWN"]
        # Normalize cached NPD IDs back to ICAO if present.
        top10 = [NPD_ID_TO_ICAO.get(t, t) for t in top10]

    doc29_root = (repo_root / args.doc29_root).resolve()
    if not doc29_root.exists():
        raise FileNotFoundError(f"Doc29 root not found: {doc29_root}")

    results_root = Path(args.results_root) if args.results_root else (repo_root / "noise_simulation" / "results" / args.experiment)
    log_path = Path(args.log_file) if args.log_file else (results_root / "run.log")
    logger = _configure_logging(log_path, args.experiment)
    _warn_missing_profiles(logger, doc29_root)
    raw_tracks_root = results_root / "raw_tracks"
    logger.info("Preparing raw tracks under %s", raw_tracks_root)
    ensure_raw_tracks(repo_root, preprocessed_path, exp_dir, raw_tracks_root)

    discrete_root = doc29_root / "Groundtracks" / args.experiment
    airport_csv = doc29_root / "Input_Airport.csv"
    summary_rows: List[Dict[str, object]] = []

    for label_path in label_files:
        flow_frames: List[tuple[str, str, pd.DataFrame]] = []
        if label_path.stem == "labels_ALL":
            labels_all = pd.read_csv(label_path)
            required = {"A/D", "Runway", "flight_id", "cluster_id"}
            if labels_all.empty or not required.issubset(labels_all.columns):
                logger.info("Skip unusable %s (missing required columns)", label_path.name)
                continue
            for (ad, runway), sub in labels_all.groupby(["A/D", "Runway"], dropna=True):
                flow_frames.append((str(ad), str(runway), sub.copy()))
        else:
            ad, runway = parse_flow_from_label(label_path)
            labels_df = pd.read_csv(label_path)
            flow_frames.append((ad, runway, labels_df))

        for ad, runway, labels_df in flow_frames:
            if runway not in doc29_inputs.RUNWAY_FILE_MAP:
                logger.info("Skip unsupported runway %s in %s", runway, label_path.name)
                continue
            if labels_df.empty or "cluster_id" not in labels_df.columns:
                continue
            labels_df = labels_df[labels_df["cluster_id"] != -1].copy()
            if labels_df.empty:
                continue
            labels_df["aircraft_type"] = labels_df["flight_id"].map(flight_type_map).fillna("UNKNOWN")

            for cluster_id, cluster_df in labels_df.groupby("cluster_id"):
                flights_by_type: Dict[str, List[int]] = {}
                for atype in top10:
                    ids = (
                        cluster_df.loc[cluster_df["aircraft_type"] == atype, "flight_id"]
                        .dropna()
                        .astype(int)
                        .unique()
                        .tolist()
                    )
                    if ids:
                        flights_by_type[atype] = ids
                if not flights_by_type:
                    continue

                mode = doc29_inputs.MODE_MAP[ad]
                gt_out_dir = discrete_root / f"{ad}_{runway}" / f"cluster_{cluster_id}" / "groundtruth"
                gt_tracks = generate_groundtruth_tracks(
                    preprocessed_path,
                    flights_by_type,
                    gt_out_dir,
                    runway=runway,
                    mode=mode,
                    interpolation_length=args.interpolation_length,
                )
                subtrack_paths = generate_discrete_tracks(
                    raw_tracks_root,
                    discrete_root,
                    args.experiment,
                    ad,
                    runway,
                    int(cluster_id),
                    args.interpolation_length,
                )

                for aircraft_type, flight_ids in flights_by_type.items():
                    suffix = _npd_suffix(ad)
                    npd_id = NPD_TYPE_MAP.get(aircraft_type, aircraft_type)
                    npd_file = doc29_root / "NPD_data" / f"{npd_id}_{suffix}.csv"
                    if not npd_file.exists():
                        logger.info("Skip missing NPD file %s", npd_file)
                        continue

                    # Keep ICAO + NPD in folder name to avoid collisions when multiple ICAO
                    # types share the same NPD ID (e.g., E170/E75L -> CF348E).
                    type_safe = _type_folder_name(aircraft_type, npd_id)
                    out_dir = results_root / f"{ad}_{runway}" / f"cluster_{cluster_id}" / type_safe
                    out_dir.mkdir(parents=True, exist_ok=True)

                    npd_rel = _relative_to_doc29(doc29_root, npd_file)
                    # Prefer aircraft-specific profiles if available.
                    height_profile = _resolve_height_profile(doc29_root, aircraft_type, ad)
                    spectral_class = _resolve_spectral_class(doc29_root, aircraft_type, ad)
                    cfg = _build_flight_cfg(
                        ad=ad,
                        runway=runway,
                        npd_table=npd_rel,
                        height_profile=height_profile,
                        spectral_class=spectral_class,
                        reference_speed=args.reference_speed,
                        engine_type=args.engine_type,
                        engine_position=args.engine_position,
                        default_startpoint=args.default_startpoint,
                    )

                    total_flights = len(flight_ids)
                    track_counts = doc29_inputs._compute_track_counts(total_flights)
                    sub_entries: List[FlightEntry] = []
                    for track_name, _weight in doc29_inputs.TRACK_DEFS:
                        rel = _relative_to_doc29(doc29_root, subtrack_paths[track_name])
                        col_name = f"Track {track_name}"
                        sub_entries.append(FlightEntry(col_name, rel, str(track_counts[track_name])))

                    sub_flight_csv = out_dir / "Flight_subtracks.csv"
                    write_flight_csv(build_columns(sub_entries, cfg), sub_flight_csv)

                    gt_entries: List[FlightEntry] = []
                    for idx, (fid, path) in enumerate(gt_tracks.get(aircraft_type, []), start=1):
                        rel = _relative_to_doc29(doc29_root, path)
                        gt_entries.append(FlightEntry(f"Flight {idx}", rel, "1"))
                    if not gt_entries:
                        logger.info("Skip missing ground truth tracks for %s in cluster %s", aircraft_type, cluster_id)
                        continue
                    gt_flight_csv = out_dir / "Flight_groundtruth.csv"
                    write_flight_csv(build_columns(gt_entries, cfg), gt_flight_csv)

                    sub_out = out_dir / "subtracks.csv"
                    gt_out = out_dir / "groundtruth.csv"
                    if not (args.skip_existing and sub_out.exists()):
                        logger.info("Running subtracks: %s", sub_out)
                        run_doc29(doc29_root, airport_csv, sub_flight_csv, args.cumulative_metric, args.time_in_s, sub_out)
                    if not (args.skip_existing and gt_out.exists()):
                        logger.info("Running groundtruth: %s", gt_out)
                        run_doc29(doc29_root, airport_csv, gt_flight_csv, args.cumulative_metric, args.time_in_s, gt_out)

                    metrics = compare_cumulative_res(sub_out, gt_out)
                    summary = {
                        "experiment": args.experiment,
                        "A/D": ad,
                        "Runway": runway,
                        "cluster_id": int(cluster_id),
                        "aircraft_type": aircraft_type,
                        "npd_id": npd_id,
                        "type_folder": type_safe,
                        "n_flights": total_flights,
                        "mse_cumulative_res": metrics["mse_cumulative_res"],
                        "mae_cumulative_res": metrics["mae_cumulative_res"],
                        "rmse_cumulative_res": metrics["rmse_cumulative_res"],
                        "n_measurements": metrics["n_measurements"],
                        "npd_table": npd_rel,
                        "height_profile": height_profile,
                        "spectral_class": spectral_class,
                        "subtracks_csv": str(sub_out),
                        "groundtruth_csv": str(gt_out),
                    }
                    (out_dir / "mse.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                    logger.info(
                        "Metrics: flow=%s/%s cluster=%s type=%s mse=%.6f mae=%.6f",
                        ad,
                        runway,
                        cluster_id,
                        aircraft_type,
                        metrics["mse_cumulative_res"],
                        metrics["mae_cumulative_res"],
                    )
                    summary_rows.append(summary)

    if summary_rows:
        summary_path = results_root / "summary_mse.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        logger.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
