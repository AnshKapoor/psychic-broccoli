from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pytz


def _detect_header_row(rows: pd.DataFrame, expected_columns: Iterable[str]) -> Optional[int]:
    """Return the row index that most closely matches the expected column names."""

    # Normalise expected column names to a comparable representation.
    expected = {col.strip().lower() for col in expected_columns}
    for idx, row in rows.iterrows():
        # Convert the row values to string for a robust comparison.
        row_values = {str(value).strip().lower() for value in row.tolist() if pd.notna(value)}
        if expected.issubset(row_values):
            return idx
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
            continue

        header_row = _detect_header_row(sheet_raw, expected_columns)
        if header_row is None:
            # No valid table was found on this sheet, so skip quietly.
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
            cleaned_frames.append(df_sheet)

    if not cleaned_frames:
        raise ValueError(
            "Unable to locate valid measurement data in any MP sheet of the workbook."
        )

    # Concatenate all measurement sheets into a single table for downstream processing.
    df_noise = pd.concat(cleaned_frames, ignore_index=True)

    return df_noise


def _load_adsb_joblib(adsb_joblib: Union[str, Path]) -> pd.DataFrame:
    """Load ADS-B data from a Joblib file and return it as a DataFrame."""

    path = Path(adsb_joblib)
    adsb_obj = joblib.load(path)

    if isinstance(adsb_obj, pd.DataFrame):
        df_ads = adsb_obj.copy()
    elif isinstance(adsb_obj, (list, tuple)):
        df_ads = pd.DataFrame(adsb_obj)
    elif isinstance(adsb_obj, dict):
        df_ads = pd.DataFrame(adsb_obj)
    else:
        raise TypeError(
            "Unsupported ADS-B Joblib payload. Expected a pandas DataFrame, "
            "a mapping, or an iterable of records."
        )

    if "timestamp" not in df_ads.columns:
        raise KeyError("ADS-B dataset does not contain the required 'timestamp' column.")

    # Ensure timestamps are timezone-aware UTC for matching operations.
    df_ads["timestamp"] = pd.to_datetime(df_ads["timestamp"], utc=True)

    return df_ads


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

    # 1) Load and tidy the noise measurements from Excel.
    df_noise = _load_noise_excel(df_noise)

    if "TLASmax_UTC" in df_noise.columns:
        # The Excel sheet already provides UTC timestamps.
        df_noise["t_ref"] = pd.to_datetime(df_noise["TLASmax_UTC"], utc=True, errors="coerce")
    else:
        # Fallback: convert the local TLASmax timestamps (Berlin time) to UTC.
        df_noise["TLASmax"] = pd.to_datetime(
            df_noise["TLASmax"], dayfirst=True, errors="coerce"
        )
        berlin_tz = pytz.timezone("Europe/Berlin")
        df_noise["t_ref"] = (
            df_noise["TLASmax"].dt.tz_localize(berlin_tz, ambiguous="infer", nonexistent="shift_forward")
            .dt.tz_convert(pytz.UTC)
        )

    # Drop rows where the timestamp could not be parsed.
    df_noise = df_noise[df_noise["t_ref"].notna()].copy()

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

    # 3) MP coordinates and bounding-box helper utilities.
    def parse_dms(dms_str: str) -> float:
        """Convert a DMS coordinate string to decimal degrees."""
        s = dms_str.replace("°"," ").replace("'"," ").replace("\""," ").strip()
        deg, minu, sec, hemi = s.split()
        dd = float(deg) + float(minu)/60 + float(sec)/3600
        return -dd if hemi in ("S","W") else dd

    mp_coords = {
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

    def get_bbox(lat0: float, lon0: float, dist_m: float, buf_frac: float) -> Tuple[float, float, float, float]:
        """Return a latitude/longitude bounding box centred at the microphone."""
        r = dist_m * (1 + buf_frac)
        deg_lat = r / 111195.0
        deg_lon = deg_lat / np.cos(np.deg2rad(lat0))
        return lat0 - deg_lat, lat0 + deg_lat, lon0 - deg_lon, lon0 + deg_lon

    # 4) Loop over noise rows, find ±tol_sec match, extract ±window_min trajectory
    tol = pd.Timedelta(seconds=tol_sec)
    win = pd.Timedelta(minutes=window_min)
    trajs = []

    for i, row in df_noise.iterrows():
        # Iterate over every noise measurement, attempting to identify a matching flight.
        t0 = row["t_ref"]
        mp = row["MP"]
        dist = row["Abstand [m]"]
        lat0, lon0 = mp_coords.get(mp, (None, None))
        if lat0 is None:
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
            continue

        # Stamp the noise row with the first matching aircraft identifiers.
        icao24, callsign = hits.iloc[0][["icao24", "callsign"]]

        df_noise.at[i, "icao24"]   = icao24
        df_noise.at[i, "callsign"] = callsign

        # extract the full ±window slice
        m2 = (
          (df_ads.icao24   == icao24) &
          (df_ads.callsign == callsign) &
          (df_ads.timestamp >= t0 - win) &
          (df_ads.timestamp <= t0 + win)
        )
        slice6 = df_ads.loc[m2].copy()
        slice6["MP"]      = mp
        slice6["t_ref"]   = t0
        slice6["icao24"]   = icao24
        slice6["callsign"] = callsign
        trajs.append(slice6)

    # 5) finalize
    df_traj = pd.concat(trajs, ignore_index=True) if trajs else pd.DataFrame()

    # 6) optional write-out
    if out_noise_csv:
        df_noise.to_csv(out_noise_csv, index=False)
    if out_traj_parquet and not df_traj.empty:
        df_traj.to_parquet(out_traj_parquet, index=False)

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

    args = parser.parse_args()

    match_noise_to_adsb(
        df_noise=args.noise_excel,
        adsb_joblib=args.adsb_joblib,
        out_noise_csv=args.noise_output,
        out_traj_parquet=args.traj_output,
        tol_sec=args.tol_sec,
        buffer_frac=args.buffer_frac,
        window_min=args.window_min,
    )


if __name__ == "__main__":
    main()
