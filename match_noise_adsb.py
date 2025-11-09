import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# ---------------------------
# Helpers for robust Excel IO
# ---------------------------

HEADER_TOKENS = {
    "mp", "ata/atd", "a/d", "runway", "sid/star", "flugzeugtyp", "mtom",
    "triebwerk", "tlasmax", "abstand", "hohenwinkel", "höhe", "lasmax",
    "leq", "lae", "t10", "tgesamt", "azb-klasse"
}


def _norm(s: Any) -> str:
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"[\u00A0\s]+", " ", s).strip().lower()
    # unify common diacritics/variants that appear in headers
    s = (s
         .replace("ö", "o").replace("ä", "a").replace("ü", "u")
         .replace("[", "").replace("]", ""))
    return s


def _looks_like_header(cells: list[str]) -> bool:
    """Heuristic: a row is a header if it contains TLAS and enough known tokens."""
    normed = [_norm(x) for x in cells]
    normed = [x for x in normed if x]
    if not normed:
        return False

    tokens = set()
    for x in normed:
        x2 = x.replace(" [m]", "")
        if "abstand" in x2:
            tokens.add("abstand")
        if "tlas" in x2:
            tokens.add("tlasmax")
        if "sid" in x2 or "star" in x2:
            tokens.add("sid/star")
        if "hohenwinkel" in x2 or "höhenwinkel" in x2:
            tokens.add("hohenwinkel")
        if x2 in HEADER_TOKENS:
            tokens.add(x2)
    return ("tlasmax" in tokens) and (len(tokens) >= 5)


def _read_noise_sheet_autoheader(excel_path: str, sheet_name: str) -> pd.DataFrame:
    """Read a sheet without assuming header row; detect it automatically."""
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, dtype=object)

    scan_rows = min(20, len(raw))
    header_idx = None
    for r in range(scan_rows):
        row_vals = raw.iloc[r].tolist()
        if _looks_like_header(row_vals):
            header_idx = r
            break
    if header_idx is None:
        header_idx = 0  # fallback

    cols = raw.iloc[header_idx].astype(str).tolist()
    cols = [re.sub(r"[\u00A0\s]+", " ", c).strip() for c in cols]

    df = raw.iloc[header_idx + 1:].copy()
    df.columns = cols

    # drop fully empty columns/rows
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    # If MP missing, backfill with sheet name
    if not any(_norm(c) == "mp" for c in df.columns):
        df["MP"] = sheet_name

    return df


# ---------------------------
# Core matching functionality
# ---------------------------

def match_noise_to_adsb(
    noise_xlsx: str,
    adsb_joblib: str,
    out_noise_csv: Optional[str] = None,
    out_traj_parquet: Optional[str] = None,
    tol_sec: int = 10,
    buffer_frac: float = 0.5,
    window_min: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match Excel noise measurements (sheets MP1..MP9; decorative rows allowed) to ADS-B trajectories loaded from a joblib file.

    Inputs
    ------
    noise_xlsx : path to Excel workbook with sheets MP1..MP9
    adsb_joblib: path to joblib containing a pandas DataFrame with ADS-B points

    Returns
    -------
    (df_noise_with_matches, concatenated_trajectory_slices)
    """

    print(f"\n📘 Reading Excel workbook (auto header detection): {noise_xlsx}")
    xls = pd.ExcelFile(noise_xlsx)
    frames = []
    for sheet in xls.sheet_names:
        df = _read_noise_sheet_autoheader(noise_xlsx, sheet)
        print(f"  • Sheet '{sheet}': detected columns -> {list(df.columns)}")
        frames.append(df)

    df_noise = pd.concat(frames, ignore_index=True)

    # Normalize MP labels like "M 01" -> "MP1"
    def _normalize_mp(x: Any) -> str:
        s = str(x)
        s = re.sub(r"[\u00A0\s]+", " ", s).strip().upper()
        s = s.replace("M ", "MP").replace("M0", "MP0").replace("M", "MP")
        s = s.replace(" ", "")
        if not s.startswith("MP"):
            s = "MP" + s
        s = re.sub(r"^MP0*([1-9]\d*)$", r"MP\1", s)  # MP01 -> MP1
        return s

    mp_candidates = [c for c in df_noise.columns if _norm(c) == "mp"]
    mp_col = mp_candidates[0] if mp_candidates else "MP"
    df_noise[mp_col] = df_noise[mp_col].apply(_normalize_mp)
    df_noise.rename(columns={mp_col: "MP"}, inplace=True)

    # Identify TLASmax column robustly
    tlas_cols = [c for c in df_noise.columns if re.search(r"tlas", _norm(c))]
    if not tlas_cols:
        tlas_col = df_noise.columns[min(8, len(df_noise.columns) - 1)]
        print(f"⚠️  No header containing 'tlas' found. Falling back to column #8: '{tlas_col}'")
    else:
        tlas_col = tlas_cols[0]
    df_noise.rename(columns={tlas_col: "TLASmax"}, inplace=True)

    # Distance column
    dist_cols = [c for c in df_noise.columns if re.search(r"abstand", _norm(c))]
    dist_col = dist_cols[0] if dist_cols else "Abstand [m]"
    df_noise["Abstand [m]"] = pd.to_numeric(df_noise.get(dist_col, np.nan), errors="coerce")

    # ------------------------------
    # Parse timestamps (DST-safe)
    # ------------------------------
    print("\n⏱️ Parsing TLASmax timestamps (DST-safe, vectorized) ...")

    # Parse TLASmax as naive datetime (no timezone yet)
    t_raw = pd.to_datetime(df_noise["TLASmax"], errors="coerce", dayfirst=True)

    # 1) Localize to Europe/Berlin
    t_loc = t_raw.dt.tz_localize(
        "Europe/Berlin",
        ambiguous="NaT",            # ambiguous fall-back hour -> NaT
        nonexistent="shift_forward"  # spring-forward gap -> shift
    )

    # 2) Retry ambiguous rows as DST (summer time)
    amb_mask = t_loc.isna() & t_raw.notna()
    if amb_mask.any():
        t_retry = t_raw[amb_mask].dt.tz_localize(
            "Europe/Berlin",
            ambiguous=True,
            nonexistent="shift_forward",
        )
        t_loc.loc[amb_mask] = t_retry

    # 3) Convert to UTC
    df_noise["t_ref"] = t_loc.dt.tz_convert("UTC")

    num_bad = df_noise["t_ref"].isna().sum()
    if num_bad:
        print(f"⚠️  {num_bad} TLASmax rows could not be parsed/localized (NaT).")

    # Prepare match fields
    df_noise["icao24"] = pd.NA
    df_noise["callsign"] = pd.NA

    # -------------------------
    # Load ADS-B (joblib DF) - *more* robust coercion & introspection
    # -------------------------
    print(f"🛩️  Loading ADS-B joblib: {adsb_joblib}")
    ads_obj: Any = joblib.load(adsb_joblib).data

    

    def _inspect(obj: Any) -> str:
        try:
            import numpy as _np
            t = type(obj)
            if isinstance(obj, dict):
                keys = list(obj.keys())[:20]
                return f"type={t.__name__}, dict_keys={keys}"
            if isinstance(obj, (list, tuple)):
                return f"type={t.__name__}, len={len(obj)}, first_type={(type(obj[0]).__name__ if obj else None)}"
            if 'DataFrame' in t.__name__:
                try:
                    # polars/dask/pandas will all show shape
                    shape = getattr(obj, 'shape', None)
                    return f"type={t.__name__}, shape={shape}"
                except Exception:
                    return f"type={t.__name__}"
            if isinstance(obj, _np.ndarray):
                return f"type=np.ndarray, shape={obj.shape}, dtype={obj.dtype}"
            return f"type={t.__name__}"
        except Exception:
            return f"type={type(obj)}"

    def _as_dataframe(obj: Any) -> pd.DataFrame:
        """Best-effort conversion of various joblib payloads into a Pandas DataFrame.
        Supported:
          - pandas.DataFrame directly
          - traffic.core.Traffic / Flight objects (use their .data DataFrame)
          - dict containing a DataFrame (common keys 'df'/'data'/'adsb'/'table', or any DF value)
          - tuple/list where first or all items are DataFrames (concatenate)
          - list of dicts -> DataFrame
          - numpy structured array / recarray -> DataFrame
          - pyarrow.Table -> to_pandas()
          - polars.DataFrame -> to_pandas()
          - dask.dataframe.DataFrame -> .compute()
          - path-like strings to parquet/csv -> read and return
        """
        # 0) traffic objects (duck-typed)
        try:
            if hasattr(obj, "data") and isinstance(getattr(obj, "data"), pd.DataFrame):
                return getattr(obj, "data")
        except Exception:
            pass

        # 1) direct pandas
        if isinstance(obj, pd.DataFrame):
            return obj

        # 2) duck types from other libs
        try:
            import pyarrow as pa  # type: ignore
            if isinstance(obj, pa.Table):
                return obj.to_pandas()
        except Exception:
            pass
        try:
            import polars as pl  # type: ignore
            if isinstance(obj, pl.DataFrame):
                return obj.to_pandas()
        except Exception:
            pass
        try:
            import dask.dataframe as dd  # type: ignore
            if isinstance(obj, dd.DataFrame):
                return obj.compute()
        except Exception:
            pass

        # 3) dict containers
        if isinstance(obj, dict):
            for k in ["df", "data", "adsb", "table"]:
                if k in obj and isinstance(obj[k], pd.DataFrame):
                    return obj[k]
            for v in obj.values():
                # nested conversions (e.g., dict holding a Traffic object)
                try:
                    return _as_dataframe(v)
                except Exception:
                    continue
            # dict of equal-length lists/arrays
            try:
                df_try = pd.DataFrame(obj)
                if not df_try.empty and df_try.columns.size > 1:
                    return df_try
            except Exception:
                pass

        # 4) list/tuple containers
        if isinstance(obj, (list, tuple)):
            # list of traffic Flight objects or any objects with .data
            try:
                dfs = []
                for x in obj:
                    if hasattr(x, "data") and isinstance(getattr(x, "data"), pd.DataFrame):
                        dfs.append(getattr(x, "data"))
                if dfs:
                    return pd.concat(dfs, ignore_index=True)
            except Exception:
                pass
            # list of DFs
            if all(isinstance(x, pd.DataFrame) for x in obj):
                return pd.concat(list(obj), ignore_index=True)
            # first element DF
            if obj and isinstance(obj[0], pd.DataFrame):
                return obj[0]
            # list of dicts
            if obj and isinstance(obj[0], dict):
                return pd.DataFrame(list(obj))
            # list/tuple -> try DataFrame
            try:
                return pd.DataFrame(obj)
            except Exception:
                pass

        # 5) numpy structured array / recarray
        try:
            import numpy as _np
            if isinstance(obj, _np.ndarray) and obj.dtype.names:
                return pd.DataFrame(obj)
        except Exception:
            pass

        # 6) path-like payloads -> read parquet/csv
        if isinstance(obj, (str, Path)):
            p = Path(obj)
            if p.suffix.lower() in {'.parquet', '.pq'} and p.exists():
                return pd.read_parquet(p)
            if p.suffix.lower() in {'.csv'} and p.exists():
                return pd.read_csv(p)

        raise TypeError("Could not coerce joblib payload to DataFrame: " + _inspect(obj))

    try:
        df_ads = _as_dataframe(ads_obj).copy()
    except Exception as e:
        print("❌ Unsupported joblib payload:", _inspect(ads_obj))
        raise TypeError(
            "ADS-B joblib should contain a table-like object. Supported: pandas/pyarrow/polars/dask; "
            "dicts/lists/tuples holding those; numpy structured arrays; or a path to parquet/csv."
            f"Details: {e}"
        )

    # Validate & normalize timestamp/geo columns
    required_cols = {"timestamp", "latitude", "longitude", "icao24", "callsign"}
    missing = [c for c in required_cols if c not in df_ads.columns]
    if missing:
        lower_map = {str(c).strip().lower(): c for c in df_ads.columns}
        for want in list(missing):
            if want in lower_map:
                df_ads.rename(columns={lower_map[want]: want}, inplace=True)
                missing.remove(want)
        if missing:
            raise ValueError(f"ADS-B DataFrame missing required columns: {missing}; columns present={list(df_ads.columns)}")

    df_ads["timestamp"] = pd.to_datetime(df_ads["timestamp"], utc=True, errors="coerce")
    df_ads = (
        df_ads.dropna(subset=["timestamp", "latitude", "longitude"])\
              .sort_values("timestamp").reset_index(drop=True)
    )

    print(f"   → ADS-B rows: {len(df_ads):,}; columns: {list(df_ads.columns)}")

    # ---------------------------------------
    # MP receiver coordinates (HAJ region)
    # ---------------------------------------
    def parse_dms(dms_str: str) -> float:
        s = dms_str.replace("°", " ").replace("'", " ").replace('"', " ").strip()
        deg, minu, sec, hemi = s.split()
        dd = float(deg) + float(minu) / 60.0 + float(sec) / 3600.0
        return -dd if hemi in ("S", "W") else dd

    mp_coords = {
        "MP1": (parse_dms('52°27\'13"N'), parse_dms('9°45\'37"E')),
        "MP2": (parse_dms('52°27\'57"N'), parse_dms('9°45\'02"E')),
        "MP3": (parse_dms('52°27\'60"N'), parse_dms('9°47\'25"E')),
        "MP4": (parse_dms('52°27\'19"N'), parse_dms('9°48\'48"E')),
        "MP5": (parse_dms('52°28\'05"N'), parse_dms('9°49\'57"E')),
        "MP6": (parse_dms('52°27\'14"N'), parse_dms('9°37\'31"E')),
        "MP7": (parse_dms('52°27\'40"N'), parse_dms('9°34\'55"E')),
        "MP8": (parse_dms('52°28\'07"N'), parse_dms('9°32\'48"E')),
        "MP9": (parse_dms('52°28\'06"N'), parse_dms('9°37\'09"E')),
    }

    def get_bbox(lat0: float, lon0: float, dist_m: float, buf_frac: float):
        if pd.isna(dist_m) or dist_m <= 0:
            dist_m = 1000.0
        r = dist_m * (1.0 + buf_frac)
        deg_lat = r / 111_195.0
        deg_lon = deg_lat / np.cos(np.deg2rad(lat0))
        return lat0 - deg_lat, lat0 + deg_lat, lon0 - deg_lon, lon0 + deg_lon

    # ------------------------------------------
    # Row-wise search + extract trajectory window
    # ------------------------------------------
    tol = pd.Timedelta(seconds=tol_sec)
    win = pd.Timedelta(minutes=window_min)
    traj_slices = []

    for i, row in df_noise.iterrows():
        t0 = row.get("t_ref", pd.NaT)
        if pd.isna(t0):
            continue
        mp = str(row.get("MP", "")).upper().replace(" ", "")
        lat0, lon0 = mp_coords.get(mp, (np.nan, np.nan))
        if pd.isna(lat0) or pd.isna(lon0):
            continue

        lat_min, lat_max, lon_min, lon_max = get_bbox(
            lat0, lon0, float(row.get("Abstand [m]", np.nan)), buffer_frac
        )

        # First time window, then spatial filter
        m_time = (df_ads["timestamp"] >= (t0 - tol)) & (df_ads["timestamp"] <= (t0 + tol))
        cand = df_ads.loc[m_time]
        if cand.empty:
            continue

        m_space = (
            (cand["latitude"] >= lat_min) & (cand["latitude"] <= lat_max) &
            (cand["longitude"] >= lon_min) & (cand["longitude"] <= lon_max)
        )
        hits = cand.loc[m_space]
        if hits.empty:
            continue

        # Choose the closest-in-time hit
        best_idx = (hits["timestamp"] - t0).abs().idxmin()
        icao24 = hits.at[best_idx, "icao24"]
        callsign = hits.at[best_idx, "callsign"]

        df_noise.at[i, "icao24"] = icao24
        df_noise.at[i, "callsign"] = callsign

        # Extract ±window slice for that aircraft/callsign
        m2 = (
            (df_ads["icao24"] == icao24) &
            (df_ads["callsign"] == callsign) &
            (df_ads["timestamp"] >= (t0 - win)) &
            (df_ads["timestamp"] <= (t0 + win))
        )
        sl = df_ads.loc[m2].copy()
        if sl.empty:
            continue
        sl["MP"] = mp
        sl["t_ref"] = t0
        traj_slices.append(sl)

    df_traj = pd.concat(traj_slices, ignore_index=True) if traj_slices else pd.DataFrame()

    # -----------------
    # Finalize / save
    # -----------------
    if out_noise_csv:
        Path(out_noise_csv).parent.mkdir(parents=True, exist_ok=True)
        df_noise.to_csv(out_noise_csv, index=False, encoding="utf-8")

    if out_traj_parquet and not df_traj.empty:
        Path(out_traj_parquet).parent.mkdir(parents=True, exist_ok=True)
        df_traj.to_parquet(out_traj_parquet, index=False)

    return df_noise, df_traj


def main():
    """Run the noise–ADS-B matching with your file paths and defaults."""
    noise_file = "noise_data.xlsx"
    adsb_file = "adsb/data_2022_april.joblib"
    out_noise = "noise_with_matches.csv"
    out_traj = "matched_trajs.parquet"

    df_noise, df_traj = match_noise_to_adsb(
        noise_xlsx=noise_file,
        adsb_joblib=adsb_file,
        out_noise_csv=out_noise,
        out_traj_parquet=out_traj,
        tol_sec=10,
        buffer_frac=1.5,
        window_min=3,
    )

    print("\n✅ Matching completed.")
    print(f"Matched noise events: {df_noise['icao24'].notna().sum()} / {len(df_noise)}")
    print(f"Trajectory slice rows: {len(df_traj)}")
    print("Outputs:")
    print(f"  • {out_noise}")
    print(f"  • {out_traj}")


if __name__ == "__main__":
    main()
