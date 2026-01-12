"""Convert ADS-B joblib files (traffic objects) to Parquet files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _coerce_to_dataframe(obj, dict_key_column: str | None = None) -> pd.DataFrame:
    """Safely coerce traffic/joblib payloads into a pandas DataFrame."""

    if isinstance(obj, pd.DataFrame):
        return obj

    if hasattr(obj, "data") and isinstance(obj.data, pd.DataFrame):
        return obj.data

    if hasattr(obj, "to_dataframe"):
        df = obj.to_dataframe()
        if isinstance(df, pd.DataFrame):
            return df

    if isinstance(obj, dict):
        frames: List[pd.DataFrame] = []
        for key, value in obj.items():
            df = None
            if isinstance(value, pd.DataFrame):
                df = value
            elif hasattr(value, "data") and isinstance(value.data, pd.DataFrame):
                df = value.data
            elif hasattr(value, "to_dataframe"):
                df = value.to_dataframe()
            if df is not None:
                if dict_key_column and dict_key_column not in df.columns:
                    df = df.copy()
                    df[dict_key_column] = key
                frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True)
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame([obj])

    if isinstance(obj, (list, tuple)):
        return pd.DataFrame(obj)

    if isinstance(obj, np.ndarray):
        return pd.DataFrame(obj)

    raise TypeError(f"Unsupported joblib payload type: {type(obj)}")


def _get_parquet_engine() -> str:
    try:
        import pyarrow  # noqa: F401

        return "pyarrow"
    except ImportError:
        try:
            import fastparquet  # noqa: F401

            return "fastparquet"
        except ImportError as exc:
            raise RuntimeError("Parquet output requires pyarrow or fastparquet.") from exc


def _write_parquet(
    df: pd.DataFrame,
    out_path: Path,
    row_group_size: int,
    compression: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine = _get_parquet_engine()
    kwargs = {"index": False, "compression": compression}
    if engine == "pyarrow" and row_group_size > 0:
        kwargs["row_group_size"] = row_group_size
    df.to_parquet(out_path, engine=engine, **kwargs)


def convert_joblib(
    joblib_path: Path,
    output_path: Path,
    row_group_size: int,
    parse_timestamp: bool,
    dict_key_column: str | None,
    compression: str,
) -> None:
    logger.info("Loading %s", joblib_path)
    obj = joblib.load(joblib_path, mmap_mode="r")
    logger.info("Payload type: %s", type(obj).__name__)
    df = _coerce_to_dataframe(obj, dict_key_column=dict_key_column)
    logger.info("Coerced to DataFrame with %d rows and %d columns", len(df), df.shape[1])

    if parse_timestamp and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    _write_parquet(df, output_path, row_group_size=row_group_size, compression=compression)
    logger.info("Wrote %s", output_path)


def _resolve_joblib_paths(joblib: str | None, input_dir: str | None, glob_pattern: str) -> List[Path]:
    if joblib:
        return [Path(joblib)]
    if not input_dir:
        raise ValueError("Provide --joblib or --input-dir.")
    base = Path(input_dir)
    return sorted(base.glob(glob_pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ADS-B joblib files to Parquet.")
    parser.add_argument("--joblib", default=None, help="Single joblib file to convert.")
    parser.add_argument("--input-dir", default="adsb", help="Directory containing joblib files.")
    parser.add_argument("--glob", default="*.joblib", help="Glob pattern for joblib files.")
    parser.add_argument("--output-dir", default="output/adsb_parquet", help="Directory for Parquet outputs.")
    parser.add_argument("--chunksize", type=int, default=200_000, help="Parquet row group size (rows).")
    parser.add_argument("--compression", default="snappy", help="Parquet compression codec.")
    parser.add_argument("--no-parse-timestamp", action="store_true", help="Disable UTC timestamp parsing.")
    parser.add_argument("--dict-key-column", default=None, help="Optional column name to preserve dict keys.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s | %(message)s")

    paths = _resolve_joblib_paths(args.joblib, args.input_dir, args.glob)
    if not paths:
        raise FileNotFoundError("No joblib files matched the input.")

    out_dir = Path(args.output_dir)
    for path in paths:
        output_path = out_dir / f"{path.stem}.parquet"
        convert_joblib(
            path,
            output_path,
            row_group_size=args.chunksize,
            parse_timestamp=not args.no_parse_timestamp,
            dict_key_column=args.dict_key_column,
            compression=args.compression,
        )


if __name__ == "__main__":
    main()
