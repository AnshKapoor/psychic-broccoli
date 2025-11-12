from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pytz
from pytz.exceptions import AmbiguousTimeError


# Configure a module-level logger that can be reused by the helper functions.
logger: logging.Logger = logging.getLogger(__name__)


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    console_level: int = logging.INFO,
) -> Path:
    """Set up application-wide logging and return the selected log file path.

    Parameters
    ----------
    log_file:
        Optional explicit location for the log file. When ``None`` the helper
        creates ``logs/python`` and generates a UTC timestamped file name.
    console_level:
        Logging level applied to the stream handler that mirrors key messages to
        stdout (defaults to :data:`logging.INFO`).

    Returns
    -------
    Path
        The fully qualified path to the log file capturing detailed execution
        information.
    """

    # Determine the log destination while creating the directory structure as needed.
    if log_file is not None:
        log_path = Path(log_file)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs") / "python"
        log_path = log_dir / f"{timestamp}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any leftover handlers from previous runs so we do not duplicate output.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    root_logger.addHandler(console_handler)

    root_logger.debug("Initialised logging. Writing details to %s", log_path)
    return log_path


def _detect_header_row(
    rows: pd.DataFrame, expected_columns: Iterable[str]
) -> Optional[int]:
    """Return the row index that most closely matches the expected column names.

    Notes
    -----
    The helper inspects each row until the expected columns are fully present.
    Logging is used to highlight which sheet rows were evaluated.
    """

    # Normalise expected column names to a comparable representation.
    expected = {col.strip().lower() for col in expected_columns}
    for idx, row in rows.iterrows():
        # Convert the row values to string for a robust comparison.
        row_values = {str(value).strip().lower() for value in row.tolist() if pd.notna(value)}
        if expected.issubset(row_values):
            logger.debug("Detected header row %s with expected columns", idx)
            return idx
    logger.debug("Failed to detect header row containing %s", expected)
    return None


def _load_noise_excel(noise_excel: Union[str, Path]) -> pd.DataFrame:
    """Load and tidy noise measurements from a workbook with multiple sheets.

    Parameters
    ----------
    noise_excel:
        Path to the formatted Excel workbook that may contain several measurement
        point sheets (e.g., ``MP1`` through ``MP9``) and possibly decorative pages.

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame containing the cleaned noise records from every
        recognised measurement-point sheet.
    """

    path = Path(noise_excel)

    logger.info("Loading noise workbook from %s", path)

    # Read every sheet without headers so we can locate the proper column row per sheet.
    raw_sheets: Dict[str, pd.DataFrame] = pd.read_excel(path, sheet_name=None, header=None)

    expected_columns: List[str] = [
        "MP",
        "TLASmax",
        "Abstand [m]",
    ]

    cleaned_frames: List[pd.DataFrame] = []

    for sheet_name, sheet_raw in raw_sheets.items():
        # Skip sheets that do not correspond to measurement points (e.g., summary pages).
        if not sheet_name.upper().startswith("MP"):
            logger.debug("Skipping non-measurement sheet %s", sheet_name)
            continue

        header_row = _detect_header_row(sheet_raw, expected_columns)
        if header_row is None:
            # No valid table was found on this sheet, so skip quietly.
            logger.warning("Sheet %s does not contain a recognised header row", sheet_name)
            continue

        df_sheet = pd.read_excel(path, sheet_name=sheet_name, header=header_row)

        # Drop fully empty rows that frequently appear in formatted Excel sheets.
        df_sheet = df_sheet.dropna(how="all").reset_index(drop=True)

        # Ensure the measurement-point column exists even if the sheet omits it explicitly.
        if "MP" not in df_sheet.columns:
            df_sheet["MP"] = sheet_name

        # Remove rows lacking essential identifiers while preserving measurement data.
        if "MP" in df_sheet.columns:
            df_sheet = df_sheet[df_sheet["MP"].notna()].copy()

        if not df_sheet.empty:
            logger.debug("Loaded %d rows from sheet %s", len(df_sheet), sheet_name)
            cleaned_frames.append(df_sheet)

    if not cleaned_frames:
        raise ValueError(
            "Unable to locate valid measurement data in any MP sheet of the workbook."
        )

    # Concatenate all measurement sheets into a single table for downstream processing.
    df_noise = pd.concat(cleaned_frames, ignore_index=True)

    logger.info("Loaded %d total noise rows", len(df_noise))

    return df_noise


def _localize_berlin(series: pd.Series) -> pd.Series:
    """Convert naive timestamps to Europe/Berlin while resolving DST transitions.

    Parameters
    ----------
    series:
        pandas Series containing timezone-naive ``Timestamp`` objects that are
        expressed in local Berlin time.

    Returns
    -------
    pd.Series
        A timezone-aware Series in the ``Europe/Berlin`` timezone where
        ambiguous timestamps (e.g., the clock rollback hour) are coerced to the
        standard-time occurrence so that downstream UTC conversion succeeds.
    """

    berlin_tz = pytz.timezone("Europe/Berlin")

    try:
        # Fast path where pandas can infer DST transitions automatically.
        return series.dt.tz_localize(
            berlin_tz, ambiguous="infer", nonexistent="shift_forward"
        )
    except AmbiguousTimeError:
        # Fall back to a manual pass if ambiguous timestamps occur (e.g. DST end).
        logger.warning("Encountered DST ambiguity; applying manual localisation fallback")
        localized = series.dt.tz_localize(
            berlin_tz, ambiguous="NaT", nonexistent="shift_forward"
        )
        ambiguous_mask = localized.isna() & series.notna()

        if ambiguous_mask.any():
            # For repeated times, assume the later (standard-time) occurrence.
            logger.debug(
                "Resolving %d ambiguous timestamps by assuming standard time",
                ambiguous_mask.sum(),
            )
            localized.loc[ambiguous_mask] = series.loc[ambiguous_mask].apply(
                lambda value: pd.Timestamp(
                    berlin_tz.localize(value.to_pydatetime(), is_dst=False)
                )
            )

        return localized


def _load_adsb_joblib(adsb_joblib: Union[str, Path]) -> pd.DataFrame:
    """Load ADS-B data from a Joblib file and normalise it to a DataFrame.

    Parameters
    ----------
    adsb_joblib:
        Path to the Joblib artefact. The payload may originate from the
        ``traffic`` library (e.g., :class:`traffic.core.Traffic` or
        :class:`traffic.core.Flight`) or be stored as a raw pandas structure.

    Returns
    -------
    pd.DataFrame
        A copy of the ADS-B samples with a mandatory ``timestamp`` column in
        UTC. Additional columns are carried over untouched for downstream
        matching.

    Raises
    ------
    TypeError
        If the Joblib payload cannot be converted into a pandas DataFrame.
    KeyError
        If the resulting DataFrame lacks the required ``timestamp`` column.
    """

    path = Path(adsb_joblib)
    logger.info("Loading ADS-B joblib payload from %s", path)

    adsb_obj: Any = joblib.load(path)
    logger.debug("ADS-B joblib payload type: %s", type(adsb_obj))

    df_ads: Optional[pd.DataFrame] = None

    if isinstance(adsb_obj, pd.DataFrame):
        # Direct pandas serialisation can be used as-is.
        df_ads = adsb_obj.copy()
    elif hasattr(adsb_obj, "data") and isinstance(getattr(adsb_obj, "data"), pd.DataFrame):
        # Traffic library objects (Traffic/Flight) expose their samples via ``.data``.
        df_ads = getattr(adsb_obj, "data").copy()
        logger.info("Extracted ADS-B records from traffic object via .data attribute")
    elif hasattr(adsb_obj, "to_dataframe"):
        # Some helpers (e.g., LazyTraffic) provide a conversion method.
        candidate = adsb_obj.to_dataframe()  # type: ignore[call-arg]
        if isinstance(candidate, pd.DataFrame):
            df_ads = candidate.copy()
            logger.info("Extracted ADS-B records via to_dataframe() helper")
    elif isinstance(adsb_obj, dict):
        df_ads = pd.DataFrame.from_dict(adsb_obj)
    elif isinstance(adsb_obj, (list, tuple)):
        df_ads = pd.DataFrame(adsb_obj)
    elif isinstance(adsb_obj, Iterable) and not isinstance(adsb_obj, (str, bytes)):
        # Fallback for generic iterables of records.
        df_ads = pd.DataFrame(list(adsb_obj))

    if df_ads is None:
        raise TypeError(
            "Unsupported ADS-B Joblib payload. Expected a pandas DataFrame, a "
            "traffic.core object exposing a .data DataFrame, or an iterable of records."
        )

    if "timestamp" not in df_ads.columns:
        raise KeyError("ADS-B dataset does not contain the required 'timestamp' column.")

    # Ensure timestamps are timezone-aware UTC for matching operations.
    df_ads["timestamp"] = pd.to_datetime(df_ads["timestamp"], utc=True)

    logger.info("Loaded %d ADS-B records", len(df_ads))

    return df_ads


def _normalise_identifier(value: Any) -> Optional[str]:
    """Convert raw ADS-B identifiers to clean uppercase strings or ``None``.

    The helper copes with mixed input types that frequently appear in the
    Joblib payloads. ``None``-like values and placeholder strings (e.g.,
    ``"nan"``) are treated as missing. Valid identifiers are stripped of
    surrounding whitespace and uppercased to provide a consistent format for
    subsequent CSV/Parquet serialisation.

    Parameters
    ----------
    value:
        Arbitrary identifier pulled from the ADS-B frame. This may be a float,
        integer, string, or even a pandas scalar object.

    Returns
    -------
    Optional[str]
        ``None`` when the identifier should be treated as missing, otherwise
        the cleaned string representation.
    """

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    # Convert pandas scalars (e.g., pd.NA) to native Python values first.
    try:
        if pd.isna(value):
            return None
    except TypeError:
        # ``pd.isna`` may raise when handed non-scalar objects; treat them as valid
        # and fall through to the generic string conversion.
        pass

    normalised: str = str(value).strip().upper()

    if normalised in {"", "NA", "NAN", "NONE"}:
        return None

    return normalised


def match_noise_to_adsb(
    df_noise: Union[str, Path],
    adsb_joblib: Union[str, Path],
    out_noise_csv: Optional[Union[str, Path]] = None,
    out_traj_parquet: Optional[Union[str, Path]] = None,
    tol_sec: int = 10,
    buffer_frac: float = 0.5,
    window_min: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match noise measurements from an Excel file to ADS-B trajectories loaded from Joblib.

    Parameters
    ----------
    df_noise:
        Path to the Excel file containing noise events. The file may include empty
        or decorative rows, which will be stripped automatically.
    adsb_joblib:
        Path to the Joblib file containing ADS-B samples. The file must provide a
        serialised pandas DataFrame or a structure convertible to one.
    out_noise_csv:
        Optional path where the annotated noise data (with matched identifiers)
        should be written as a CSV file.
    out_traj_parquet:
        Optional path for writing the extracted ADS-B trajectory snippets.
    tol_sec:
        Time tolerance in seconds used when searching for an ADS-B hit around the
        noise measurement timestamp.
    buffer_frac:
        Fractional buffer applied to the lateral distance filter. This widens the
        spatial bounding box by ``1 + buffer_frac``.
    window_min:
        Size of the time window in minutes to extract the matched trajectory
        segment around the reference timestamp.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the enriched noise DataFrame and the concatenated
        trajectory slices.
    """

    logger.info("Starting noise to ADS-B matching workflow")

    # 1) Load and tidy the noise measurements from Excel.
    df_noise = _load_noise_excel(df_noise)
    logger.debug("Noise dataframe columns: %s", df_noise.columns.tolist())

    if "TLASmax_UTC" in df_noise.columns:
        # The Excel sheet already provides UTC timestamps.
        df_noise["t_ref"] = pd.to_datetime(df_noise["TLASmax_UTC"], utc=True, errors="coerce")
        logger.info("Detected pre-computed UTC timestamps in TLASmax_UTC column")
    else:
        # Fallback: convert the local TLASmax timestamps (Berlin time) to UTC.
        df_noise["TLASmax"] = pd.to_datetime(
            df_noise["TLASmax"], dayfirst=True, errors="coerce"
        )
        localized = _localize_berlin(df_noise["TLASmax"])
        df_noise["t_ref"] = localized.dt.tz_convert(pytz.UTC)
        logger.info("Converted TLASmax values from local Berlin time to UTC")

    # Drop rows where the timestamp could not be parsed.
    df_noise = df_noise[df_noise["t_ref"].notna()].copy()
    logger.info("Retained %d noise rows after timestamp parsing", len(df_noise))

    if "Abstand [m]" not in df_noise.columns:
        raise KeyError("Noise dataset does not contain the required 'Abstand [m]' column.")

    if "MP" not in df_noise.columns:
        raise KeyError("Noise dataset does not contain the required 'MP' column.")

    # Ensure the distance column can be used numerically even if commas are used.
    df_noise["Abstand [m]"] = pd.to_numeric(
        df_noise["Abstand [m]"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    df_noise = df_noise[df_noise["Abstand [m]"].notna()].copy()

    df_noise["MP"] = (
        df_noise["MP"].astype(str).str.strip().str.upper().str.replace(" ", "")
    )

    # Map potential short MP identifiers (e.g., MP1) to the dictionary keys (M01).
    df_noise["MP"] = df_noise["MP"].str.replace("MP", "M", regex=False)
    df_noise["MP"] = df_noise["MP"].str.replace(r"^(M)(\d)$", r"\g<1>0\g<2>", regex=True)

    # Prepare output columns that will hold the matched flight identifiers.
    df_noise["icao24"] = pd.NA
    df_noise["callsign"] = pd.NA

    # 2) Load ADS-B data from Joblib and ensure timestamps are in UTC.
    df_ads = _load_adsb_joblib(adsb_joblib)
    logger.debug("ADS-B dataframe columns: %s", df_ads.columns.tolist())

    # 3) MP coordinates and bounding-box helper utilities.
    def parse_dms(dms_str: str) -> float:
        """Convert a DMS coordinate string (e.g. ``52°27'13"N``) into decimal degrees."""

        # Replace compass markers with space separators and split into components.
        sanitized = (
            dms_str.replace("°", " ").replace("'", " ").replace("\"", " ").strip()
        )
        deg_str, min_str, sec_str, hemisphere = sanitized.split()
        degrees = float(deg_str) + float(min_str) / 60 + float(sec_str) / 3600
        return -degrees if hemisphere in ("S", "W") else degrees

    mp_coords: Dict[str, Tuple[float, float]] = {
        "M01": (parse_dms("52°27'13\"N"), parse_dms("9°45'37\"E")),
        "M02": (parse_dms("52°27'57\"N"), parse_dms("9°45'02\"E")),
        "M03": (parse_dms("52°27'60\"N"), parse_dms("9°47'25\"E")),
        "M04": (parse_dms("52°27'19\"N"), parse_dms("9°48'48\"E")),
        "M05": (parse_dms("52°28'05\"N"), parse_dms("9°49'57\"E")),
        "M06": (parse_dms("52°27'14\"N"), parse_dms("9°37'31\"E")),
        "M07": (parse_dms("52°27'40\"N"), parse_dms("9°34'55\"E")),
        "M08": (parse_dms("52°28'07\"N"), parse_dms("9°32'48\"E")),
        "M09": (parse_dms("52°28'06\"N"), parse_dms("9°37'09\"E")),
    }

    def get_bbox(
        lat0: float,
        lon0: float,
        dist_m: float,
        buf_frac: float,
    ) -> Tuple[float, float, float, float]:
        """Return a latitude/longitude bounding box centred at the microphone."""

        radius = dist_m * (1 + buf_frac)
        deg_lat = radius / 111_195.0
        deg_lon = deg_lat / np.cos(np.deg2rad(lat0))
        return lat0 - deg_lat, lat0 + deg_lat, lon0 - deg_lon, lon0 + deg_lon

    # 4) Loop over noise rows, find ±tol_sec match, extract ±window_min trajectory
    tol = pd.Timedelta(seconds=tol_sec)
    win = pd.Timedelta(minutes=window_min)
    trajs: List[pd.DataFrame] = []

    for i, row in df_noise.iterrows():
        # Iterate over every noise measurement, attempting to identify a matching flight.
        t0 = row["t_ref"]
        mp = row["MP"]
        dist = row["Abstand [m]"]
        lat0, lon0 = mp_coords.get(mp, (None, None))
        if lat0 is None:
            logger.warning("Measurement point %s missing from coordinate map", mp)
            continue

        # spatial + time mask to identify flight
        lat_min, lat_max, lon_min, lon_max = get_bbox(lat0, lon0, dist, buffer_frac)
        m1 = (
          (df_ads.timestamp >=  t0 - tol) &
          (df_ads.timestamp <=  t0 + tol) &
          (df_ads.latitude  >= lat_min) &
          (df_ads.latitude  <= lat_max) &
          (df_ads.longitude >= lon_min) &
          (df_ads.longitude <= lon_max)
        )
        hits = df_ads.loc[m1]
        if hits.empty:
            logger.debug("No ADS-B hits found for MP %s at %s", mp, t0)
            continue

        # Stamp the noise row with the first matching aircraft identifiers.
        raw_icao24, raw_callsign = hits.iloc[0][["icao24", "callsign"]]
        icao24_clean = _normalise_identifier(raw_icao24)
        callsign_clean = _normalise_identifier(raw_callsign)

        if icao24_clean is None and callsign_clean is None:
            logger.debug(
                "Skipping assignment for MP %s at %s due to missing identifiers",
                mp,
                t0,
            )
            continue

        df_noise.at[i, "icao24"] = icao24_clean if icao24_clean is not None else pd.NA
        df_noise.at[i, "callsign"] = (
            callsign_clean if callsign_clean is not None else pd.NA
        )
        logger.info(
            "Matched MP %s at %s to aircraft %s (%s)",
            mp,
            t0,
            callsign_clean,
            icao24_clean,
        )

        # extract the full ±window slice
        m2 = (
          (df_ads.timestamp >= t0 - win) &
          (df_ads.timestamp <= t0 + win)
        )
        if icao24_clean is not None:
            m2 &= df_ads.icao24 == raw_icao24
        if callsign_clean is not None:
            m2 &= df_ads.callsign == raw_callsign
        slice6 = df_ads.loc[m2].copy()
        slice6["MP"]      = mp
        slice6["t_ref"]   = t0
        slice6["icao24"] = icao24_clean if icao24_clean is not None else pd.NA
        slice6["callsign"] = (
            callsign_clean if callsign_clean is not None else pd.NA
        )
        trajs.append(slice6)

    # 5) finalize
    for column_name in ("icao24", "callsign"):
        if column_name in df_noise.columns:
            # Enforce pandas' native string dtype so to_csv serialises the values
            # instead of silently dropping them when mixed with missing entries.
            df_noise[column_name] = df_noise[column_name].astype("string")

    df_traj = pd.concat(trajs, ignore_index=True) if trajs else pd.DataFrame()
    if not df_traj.empty:
        for column_name in ("icao24", "callsign"):
            if column_name in df_traj.columns:
                df_traj[column_name] = df_traj[column_name].astype("string")
    logger.info("Constructed %d trajectory slices", len(trajs))

    matched_rows = int(df_noise["icao24"].notna().sum())
    logger.info("Matched %d noise rows out of %d", matched_rows, len(df_noise))

    # 6) optional write-out
    if out_noise_csv:
        out_noise_path = Path(out_noise_csv)
        out_noise_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing annotated noise data to %s", out_noise_path)
        df_noise.to_csv(out_noise_path, index=False)
    if out_traj_parquet and not df_traj.empty:
        out_traj_path = Path(out_traj_parquet)
        out_traj_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing matched trajectory data to %s", out_traj_path)
        df_traj.to_parquet(out_traj_path, index=False)
    elif out_traj_parquet:
        logger.info("No trajectory data to write for %s", out_traj_parquet)

    logger.info("Noise to ADS-B matching completed")

    return df_noise, df_traj


def main() -> None:
    """Parse command-line arguments and execute the ADS-B/noise matching workflow."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Match noise measurements from Excel with ADS-B Joblib data."
    )
    parser.add_argument("noise_excel", type=Path, help="Path to the noise Excel file.")
    parser.add_argument("adsb_joblib", type=Path, help="Path to the ADS-B Joblib file.")
    parser.add_argument(
        "--noise-output",
        type=Path,
        default=None,
        help="Optional CSV output path for the annotated noise data.",
    )
    parser.add_argument(
        "--traj-output",
        type=Path,
        default=None,
        help="Optional Parquet output path for the extracted trajectories.",
    )
    parser.add_argument(
        "--tol-sec",
        type=int,
        default=10,
        help="Time tolerance in seconds for matching noise to ADS-B records.",
    )
    parser.add_argument(
        "--buffer-frac",
        type=float,
        default=0.5,
        help="Fractional buffer applied to the spatial bounding box.",
    )
    parser.add_argument(
        "--window-min",
        type=int,
        default=3,
        help="Time window in minutes to extract around the reference timestamp.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Console logging level (e.g. DEBUG, INFO, WARNING).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit log file path. Defaults to logs/python/<timestamp>.log"
        ),
    )

    args = parser.parse_args()

    console_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_path = setup_logging(args.log_file, console_level=console_level)
    logger.info("Writing detailed log to %s", log_path)
    logger.info("Console log level set to %s", logging.getLevelName(console_level))

    match_noise_to_adsb(
        df_noise=args.noise_excel,
        adsb_joblib=args.adsb_joblib,
        out_noise_csv=args.noise_output,
        out_traj_parquet=args.traj_output,
        tol_sec=args.tol_sec,
        buffer_frac=args.buffer_frac,
        window_min=args.window_min,
    )

    logger.info("Merge process finished successfully")


if __name__ == "__main__":
    main()
