"""EDA for ADS-B monthly joblib files.

Usage:
  python scripts/eda_adsb_monthly.py --input-dir adsb --outdir output/eda/adsb_monthly
"""
from __future__ import annotations

import argparse
from pathlib import Path
import logging
from typing import Any

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_adsb(path: Path) -> pd.DataFrame:
    obj: Any = joblib.load(path)
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "data") and isinstance(getattr(obj, "data"), pd.DataFrame):
        return getattr(obj, "data").copy()
    if hasattr(obj, "to_dataframe"):
        df = obj.to_dataframe()
        if isinstance(df, pd.DataFrame):
            return df.copy()
    if isinstance(obj, dict):
        return pd.DataFrame.from_dict(obj)
    if isinstance(obj, (list, tuple)):
        return pd.DataFrame(obj)
    raise TypeError(f"Unsupported payload type: {type(obj)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA summary for ADS-B monthly joblib files.")
    parser.add_argument("--input-dir", default="adsb", help="Directory with monthly joblib files.")
    parser.add_argument("--outdir", default="output/eda/adsb_monthly", help="Output directory for CSV/plots.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("logs") / "eda"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "eda_adsb_monthly.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    rows = []
    top_icao_rows = []

    files = sorted(input_dir.glob("*.joblib"))
    if not files:
        raise FileNotFoundError(f"No joblib files found in {input_dir}")

    for path in files:
        logging.info("Processing %s", path.name)
        df = _load_adsb(path)
        row = {
            "file": path.name,
            "rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "columns": ", ".join(map(str, df.columns)),
        }

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            row["timestamp_min_utc"] = str(ts.min())
            row["timestamp_max_utc"] = str(ts.max())
        else:
            row["timestamp_min_utc"] = ""
            row["timestamp_max_utc"] = ""

        if "icao24" in df.columns:
            row["unique_icao24"] = int(df["icao24"].nunique(dropna=True))
            counts = df["icao24"].value_counts(dropna=True).head(10)
            for icao, cnt in counts.items():
                top_icao_rows.append({"file": path.name, "icao24": str(icao), "count": int(cnt)})
        else:
            row["unique_icao24"] = 0

        if "callsign" in df.columns:
            row["unique_callsign"] = int(df["callsign"].nunique(dropna=True))
        else:
            row["unique_callsign"] = 0

        # If an aircraft type column exists, include it
        type_col = None
        for col in ("aircraft_type", "typecode", "aircraft_type_adsb"):
            if col in df.columns:
                type_col = col
                break
        if type_col:
            row["unique_aircraft_type"] = int(df[type_col].nunique(dropna=True))
        else:
            row["unique_aircraft_type"] = 0

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "adsb_monthly_summary.csv", index=False)

    if top_icao_rows:
        pd.DataFrame(top_icao_rows).to_csv(outdir / "adsb_monthly_top_icao24.csv", index=False)

    # Simple plots
    plt.style.use("seaborn-v0_8-whitegrid")
    def _month_label(filename: str) -> str:
        name = filename.replace("data_", "").replace(".joblib", "")
        # Expected: 2022_april
        parts = name.split("_", 1)
        if len(parts) == 2:
            return parts[1].capitalize()
        return name.capitalize()

    month_labels = [_month_label(f) for f in summary["file"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(month_labels, summary["rows"], color="#4E79A7", label="2022")
    ax.set_title("ADS-B rows per month")
    ax.set_ylabel("rows")
    ax.set_xlabel("month")
    ax.legend(title="Year", loc="best")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "rows_by_month.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["file"], summary["unique_icao24"], color="#59A14F")
    ax.set_title("Unique ICAO24 per monthly file")
    ax.set_ylabel("unique icao24")
    ax.set_xlabel("file")
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "unique_icao24_by_month.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["file"], summary["unique_callsign"], color="#F28E2B")
    ax.set_title("Unique callsigns per monthly file")
    ax.set_ylabel("unique callsigns")
    ax.set_xlabel("file")
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "unique_callsign_by_month.png", dpi=200)
    plt.close(fig)

    print(f"Wrote summary to {outdir / 'adsb_monthly_summary.csv'}")
    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()
