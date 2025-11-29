from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from pytz.exceptions import AmbiguousTimeError


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure basic console logging."""

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(message)s",
    )
    logger.debug("Logging configured at level %s", logging.getLevelName(level))


def _detect_header_row(rows: pd.DataFrame, expected_columns: Iterable[str]) -> Optional[int]:
    """Return the row index that most closely matches the expected column names."""

    expected = {col.strip().lower() for col in expected_columns}
    for idx, row in rows.iterrows():
        row_values = {str(value).strip().lower() for value in row.tolist() if pd.notna(value)}
        if expected.issubset(row_values):
            return idx
    return None


def _localize_berlin(series: pd.Series) -> pd.Series:
    """Convert naive timestamps to Europe/Berlin while handling DST quirks."""

    berlin_tz = pytz.timezone("Europe/Berlin")
    try:
        return series.dt.tz_localize(berlin_tz, ambiguous="infer", nonexistent="shift_forward")
    except AmbiguousTimeError:
        localized = series.dt.tz_localize(
            berlin_tz, ambiguous="NaT", nonexistent="shift_forward"
        ).copy()
        ambiguous_mask = localized.isna() & series.notna()
        if ambiguous_mask.any():
            localized.loc[ambiguous_mask] = series.loc[ambiguous_mask].apply(
                lambda value: pd.Timestamp(berlin_tz.localize(value.to_pydatetime(), is_dst=False))
            )
        return localized


def _normalise_mp(value: object) -> Optional[str]:
    """Return a normalised measurement-point identifier such as ``M01``."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().upper().replace(" ", "")
    if not text:
        return None
    if text.startswith("MP"):
        text = text.replace("MP", "M", 1)
    text = re.sub(r"^M(\d)$", r"M0\1", text)
    return text


def load_noise_metadata(noise_excel: Path) -> pd.DataFrame:
    """Load the noise Excel workbook and return key metadata per measurement."""

    logger.info("Loading noise workbook from %s", noise_excel)
    raw_sheets = pd.read_excel(noise_excel, sheet_name=None, header=None)
    expected_columns = ["MP", "TLASmax", "Abstand [m]"]

    frames = []
    for sheet_name, sheet_raw in raw_sheets.items():
        if not str(sheet_name).upper().startswith("MP"):
            logger.debug("Skipping non-MP sheet %s", sheet_name)
            continue

        header_row = _detect_header_row(sheet_raw, expected_columns)
        if header_row is None:
            logger.warning("Unable to detect header row for sheet %s", sheet_name)
            continue

        df_sheet = pd.read_excel(noise_excel, sheet_name=sheet_name, header=header_row)
        df_sheet = df_sheet.dropna(how="all").reset_index(drop=True)

        if "MP" not in df_sheet.columns:
            df_sheet["MP"] = sheet_name
        df_sheet = df_sheet[df_sheet["MP"].notna()].copy()
        df_sheet["MP_norm"] = df_sheet["MP"].apply(_normalise_mp)

        df_sheet["TLASmax"] = pd.to_datetime(df_sheet.get("TLASmax"), dayfirst=True, errors="coerce")
        df_sheet["ATA/ATD"] = pd.to_datetime(df_sheet.get("ATA/ATD"), dayfirst=True, errors="coerce")

        df_sheet["t_ref"] = _localize_berlin(df_sheet["TLASmax"]).dt.tz_convert(pytz.UTC)
        df_sheet["ata_atd_utc"] = _localize_berlin(df_sheet["ATA/ATD"]).dt.tz_convert(pytz.UTC)

        keep_cols = ["MP_norm", "t_ref", "ata_atd_utc", "A/D", "Runway"]
        frames.append(df_sheet[keep_cols])

    if not frames:
        raise ValueError("No measurement-point sheets with expected columns were found.")

    df_noise = pd.concat(frames, ignore_index=True)
    df_noise = df_noise[df_noise["MP_norm"].notna() & df_noise["t_ref"].notna()].copy()
    df_noise["t_ref_round"] = df_noise["t_ref"].dt.round("s")
    df_noise["ata_round"] = df_noise["ata_atd_utc"].dt.round("s")

    logger.info("Loaded %d noise measurements with timing metadata", len(df_noise))
    return df_noise


def _find_noise_match(
    df_noise: pd.DataFrame,
    mp_norm: Optional[str],
    event_time: Optional[pd.Timestamp],
    hour_hint: Optional[pd.Timestamp],
    tolerance: pd.Timedelta,
) -> Optional[Tuple[str, str]]:
    """Return the (A/D, Runway) pair that best matches the MP and time constraints."""

    if mp_norm is None:
        return None

    candidates = df_noise[df_noise["MP_norm"] == mp_norm]
    if candidates.empty:
        return None

    if event_time is not None and not pd.isna(event_time):
        delta = (candidates["t_ref_round"] - event_time.round("s")).abs()
        within = candidates.loc[delta <= tolerance]
        if not within.empty:
            best = within.loc[delta.loc[within.index].idxmin()]
            return best["A/D"], best["Runway"]

        delta_ata = (candidates["ata_round"] - event_time.round("s")).abs()
        within_ata = candidates.loc[delta_ata <= tolerance]
        if not within_ata.empty:
            best = within_ata.loc[delta_ata.loc[within_ata.index].idxmin()]
            return best["A/D"], best["Runway"]

    if hour_hint is not None and not pd.isna(hour_hint):
        hour_floor = hour_hint.floor("H")
        same_hour = candidates[candidates["t_ref_round"].dt.floor("H") == hour_floor]
        if not same_hour.empty:
            best = same_hour.iloc[0]
            return best["A/D"], best["Runway"]

    return None


def enrich_csv_with_noise(
    input_csv: Path,
    noise_df: pd.DataFrame,
    tolerance: pd.Timedelta,
    test_mode: bool,
    test_match_limit: int,
) -> pd.DataFrame:
    """Append A/D and Runway columns to a matched trajectory CSV."""

    logger.info("Reading matched trajectory CSV %s", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)

    df["MP_norm"] = df["MP"].apply(_normalise_mp)
    df["t_ref_dt"] = pd.to_datetime(df.get("t_ref"), utc=True, errors="coerce")
    df["hour_dt"] = pd.to_datetime(df.get("hour"), utc=True, errors="coerce")
    df["timestamp_dt"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")

    df["event_time"] = df["t_ref_dt"].combine_first(df["timestamp_dt"]).combine_first(df["hour_dt"])
    df["event_time"] = df["event_time"].dt.round("s")

    unique_events = df[["MP_norm", "event_time", "hour_dt"]].drop_duplicates()

    mapping: Dict[Tuple[str, pd.Timestamp], Tuple[object, object]] = {}
    matches_found = 0

    for row in unique_events.itertuples(index=False):
        mp_norm = row.MP_norm
        event_time = row.event_time
        hour_hint = row.hour_dt
        match = _find_noise_match(noise_df, mp_norm, event_time, hour_hint, tolerance)
        if match is None:
            continue
        mapping[(mp_norm, event_time)] = match
        matches_found += 1
        if test_mode and matches_found >= test_match_limit:
            logger.info("Test mode active: collected %d matches, stopping early", matches_found)
            break

    if not mapping:
        logger.warning("No matches found for %s", input_csv)
        df["A/D"] = pd.NA
        df["Runway"] = pd.NA
    else:
        df["match_key"] = list(zip(df["MP_norm"], df["event_time"]))
        df["A/D"] = df["match_key"].map(lambda key: mapping.get(key, (pd.NA, pd.NA))[0])
        df["Runway"] = df["match_key"].map(lambda key: mapping.get(key, (pd.NA, pd.NA))[1])

        if test_mode:
            allowed_keys = set(mapping.keys())
            df = df[df["match_key"].isin(allowed_keys)].copy()

    df.drop(columns=["MP_norm", "t_ref_dt", "hour_dt", "timestamp_dt", "event_time", "match_key"], errors="ignore", inplace=True)
    return df


def build_output_path(input_csv: Path, output_dir: Optional[Path], suffix: str) -> Path:
    """Return the output path for a given input CSV."""

    if output_dir is not None:
        return output_dir / f"{input_csv.stem}{suffix}{input_csv.suffix}"
    return input_csv.with_name(f"{input_csv.stem}{suffix}{input_csv.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add A/D and Runway columns from the noise Excel workbook to existing "
            "matched ADS-B trajectory CSV files."
        )
    )
    parser.add_argument("noise_excel", type=Path, help="Path to noise_data.xlsx")
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="+",
        help="One or more matched trajectory CSV files to enrich.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for enriched CSV files. Defaults to alongside each input file.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_with_ad_runway",
        help="Filename suffix inserted before .csv when output-dir is not provided.",
    )
    parser.add_argument(
        "--time-tolerance-sec",
        type=int,
        default=30,
        help="Maximum allowed time difference (in seconds) when matching to noise rows.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Console logging level (e.g. DEBUG, INFO).",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Process only the first few matches so the workflow can be smoke-tested quickly.",
    )
    parser.add_argument(
        "--test-match-limit",
        type=int,
        default=5,
        help="Number of matched events to keep when test mode is active.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    tolerance = pd.Timedelta(seconds=args.time_tolerance_sec)
    noise_df = load_noise_metadata(args.noise_excel)

    for input_csv in args.input_csv:
        output_path = build_output_path(input_csv, args.output_dir, args.suffix)
        enriched = enrich_csv_with_noise(
            input_csv=input_csv,
            noise_df=noise_df,
            tolerance=tolerance,
            test_mode=args.test_mode,
            test_match_limit=args.test_match_limit,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(output_path, index=False)
        logger.info("Wrote enriched CSV to %s (%d rows)", output_path, len(enriched))


if __name__ == "__main__":
    main()
