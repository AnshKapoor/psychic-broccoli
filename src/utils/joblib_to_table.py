import joblib
import pandas as pd
import numpy as np
from pathlib import Path


def _coerce_to_dataframe(obj) -> pd.DataFrame:
    """
    Safely coerce traffic/joblib payloads into a pandas DataFrame.
    """
    if isinstance(obj, pd.DataFrame):
        return obj

    if hasattr(obj, "data") and isinstance(obj.data, pd.DataFrame):
        return obj.data

    if hasattr(obj, "to_dataframe"):
        df = obj.to_dataframe()
        if isinstance(df, pd.DataFrame):
            return df

    if isinstance(obj, dict):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame([obj])

    if isinstance(obj, (list, tuple)):
        return pd.DataFrame(obj)

    if isinstance(obj, np.ndarray):
        return pd.DataFrame(obj)

    raise TypeError(f"Unsupported joblib payload type: {type(obj)}")


def joblib_to_table(
    joblib_path,
    out_path=None,
    out_format=None,
    chunksize=100_000,       # prevent RAM explosion
    parse_timestamp=True,
):
    """
    Memory-safe conversion of ADS-B joblib to CSV/Parquet.
    - Uses joblib memory mapping
    - Uses chunked CSV writing (prevents memory exceeded)
    - Parquet writing supported for smaller files
    """
    joblib_path = Path(joblib_path)

    # --------------------------
    # Memory-mapped loading
    # --------------------------
    obj = joblib.load(joblib_path, mmap_mode="r")
    df = _coerce_to_dataframe(obj)

    # --------------------------
    # Timestamp handling
    # --------------------------
    if parse_timestamp and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # --------------------------
    # No saving → return DF
    # --------------------------
    if out_path is None:
        return df

    out_path = Path(out_path)
    if out_format is None:
        out_format = out_path.suffix.lower().lstrip(".")

    # --------------------------
    # Parquet (not chunked but efficient)
    # --------------------------
    if out_format == "parquet":
        df.to_parquet(out_path, index=False)
        return df

    # --------------------------
    # Chunked CSV writing (SAFE)
    # --------------------------
    if out_format == "csv":
        # Write header first
        df.iloc[:0].to_csv(out_path, index=False)

        # Append chunk by chunk
        for start in range(0, len(df), chunksize):
            chunk = df.iloc[start : start + chunksize]
            chunk.to_csv(out_path, mode="a", header=False, index=False)

        return df

    else:
        raise ValueError(f"Unsupported output format: {out_format}")
