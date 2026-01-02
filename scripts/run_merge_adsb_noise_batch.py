"""Batch runner for merge_adsb_noise using a YAML config."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from merge_adsb_noise import match_noise_to_adsb, setup_logging


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _parse_year_month(stem: str) -> Tuple[str | None, str | None]:
    match = re.search(r"(?P<year>\d{4})_(?P<month>[A-Za-z]+)$", stem)
    if not match:
        return None, None
    return match.group("year"), match.group("month").lower()


def _resolve_joblibs(adsb_cfg: Dict[str, Any]) -> List[Path]:
    explicit: List[str] = list(adsb_cfg.get("joblibs") or [])
    if explicit:
        return [Path(p) for p in explicit]

    pattern = str(adsb_cfg.get("joblib_glob") or "")
    if not pattern:
        raise ValueError("Config must provide adsb.joblib_glob or adsb.joblibs")
    return sorted(Path().glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run merge_adsb_noise for multiple months via YAML config.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/merge_adsb_noise.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))

    log_cfg = cfg.get("logging", {}) or {}
    level_name = str(log_cfg.get("level", "INFO")).upper()
    console_level = getattr(logging, level_name, logging.INFO)
    setup_logging(log_cfg.get("log_file"), console_level=console_level)

    noise_excel = Path(cfg.get("noise_excel", "noise_data.xlsx"))
    joblibs = _resolve_joblibs(cfg.get("adsb", {}) or {})
    if not joblibs:
        raise FileNotFoundError("No ADS-B joblib files matched the provided configuration.")

    out_cfg = cfg.get("output", {}) or {}
    template = str(out_cfg.get("traj_output_template", "matched_trajs_{month}_{year}.parquet"))
    output_dir = out_cfg.get("output_dir")

    match_cfg = cfg.get("matching", {}) or {}
    max_km = float(match_cfg.get("max_airport_distance_km", 12))
    max_airport_distance_m = max_km * 1000.0

    testing_cfg = cfg.get("testing", {}) or {}
    test_mode = bool(testing_cfg.get("enabled", False))
    test_limit = int(testing_cfg.get("match_limit", 5))

    for adsb_joblib in joblibs:
        stem = adsb_joblib.stem
        year, month = _parse_year_month(stem)
        out_name = template.format(
            year=year or "",
            month=month or stem,
            adsb_stem=stem,
        )
        logger = logging.getLogger(__name__)
        logger.info("Processing %s -> %s", adsb_joblib, out_name)

        match_noise_to_adsb(
            df_noise=noise_excel,
            adsb_joblib=adsb_joblib,
            out_traj_parquet=out_name,
            output_dir=output_dir,
            tol_sec=int(match_cfg.get("tol_sec", 10)),
            buffer_frac=float(match_cfg.get("buffer_frac", 0.5)),
            window_min=int(match_cfg.get("window_min", 3)),
            sample_interval_sec=int(match_cfg.get("sample_interval_sec", 2)),
            max_airport_distance_m=max_airport_distance_m,
            dedupe_traj=bool(match_cfg.get("dedupe_traj", True)),
            test_mode=test_mode,
            test_mode_match_limit=test_limit,
        )


if __name__ == "__main__":
    main()
