"""
EDA for UTM-based trajectory CSVs.

Usage:
  python scripts/eda_utm.py ^
    --input data/preprocessed/preprocessed_1.csv ^
    --outdir output/eda/utm/preprocessed_1 ^
    --sample-total 200 ^
    --sample-per-flow 30 ^
    --seed 11

Input columns expected:
  step, x_utm, y_utm, A/D, Runway, flight_id, aircraft_type_match

Outputs:
  - counts_by_flow.csv
  - counts_by_aircraft_type.csv
  - flight_metrics.csv
  - flight_outliers.csv
  - summary.json
  - plots (*.png)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class EdaConfig:
    input_csv: Path
    outdir: Path
    sample_total: int
    sample_per_flow: int
    seed: int
    max_rows: int | None = None


NEEDED_COLS = [
    "step",
    "x_utm",
    "y_utm",
    "A/D",
    "Runway",
    "flight_id",
    "aircraft_type_match",
]


def _read_csv(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in NEEDED_COLS, nrows=max_rows)
    missing = [c for c in NEEDED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def _compute_flight_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(["flight_id", "step"]).reset_index(drop=True)
    dx = df_sorted.groupby("flight_id")["x_utm"].diff()
    dy = df_sorted.groupby("flight_id")["y_utm"].diff()
    seg = np.sqrt(dx * dx + dy * dy)

    lengths = seg.groupby(df_sorted["flight_id"]).sum().fillna(0.0)
    n_points = df_sorted.groupby("flight_id").size()

    first = df_sorted.groupby("flight_id")[["x_utm", "y_utm"]].first()
    last = df_sorted.groupby("flight_id")[["x_utm", "y_utm"]].last()
    displacement = np.sqrt(((last - first) ** 2).sum(axis=1))

    meta = df_sorted.groupby("flight_id").first()[["A/D", "Runway", "aircraft_type_match"]]
    metrics = pd.DataFrame(
        {
            "flight_id": lengths.index,
            "n_points": n_points.values,
            "length_m": lengths.values,
            "displacement_m": displacement.values,
        }
    ).set_index("flight_id")
    metrics = metrics.join(meta)
    metrics = metrics.reset_index()

    # Simple segment statistics (median and p95) per flight
    seg_stats = seg.groupby(df_sorted["flight_id"]).agg(
        seg_median="median",
        seg_p95=lambda s: float(np.quantile(s.dropna(), 0.95)) if len(s.dropna()) else 0.0,
    )
    metrics = metrics.join(seg_stats, on="flight_id")

    # Length / displacement ratio (curviness proxy)
    metrics["length_disp_ratio"] = metrics["length_m"] / metrics["displacement_m"].replace(0.0, np.nan)
    metrics["length_disp_ratio"] = metrics["length_disp_ratio"].fillna(0.0)
    return metrics


def _write_counts(metrics: pd.DataFrame, outdir: Path) -> Dict[str, int]:
    counts_by_flow = (
        metrics.groupby(["A/D", "Runway"], dropna=False)
        .size()
        .reset_index(name="n_flights")
        .sort_values(["A/D", "Runway"])
    )
    counts_by_flow.to_csv(outdir / "counts_by_flow.csv", index=False)

    counts_by_aircraft = (
        metrics.groupby("aircraft_type_match", dropna=False)
        .size()
        .reset_index(name="n_flights")
        .sort_values("n_flights", ascending=False)
    )
    counts_by_aircraft.to_csv(outdir / "counts_by_aircraft_type.csv", index=False)

    return {
        "total_flights": int(len(metrics)),
        "unique_flows": int(counts_by_flow.shape[0]),
        "unique_aircraft_types": int(counts_by_aircraft.shape[0]),
    }


def _write_outliers(metrics: pd.DataFrame, outdir: Path) -> Dict[str, float]:
    q_low = metrics["length_m"].quantile(0.05)
    q_high = metrics["length_m"].quantile(0.95)
    ratio_high = metrics["length_disp_ratio"].quantile(0.95)

    outliers = metrics[
        (metrics["length_m"] <= q_low)
        | (metrics["length_m"] >= q_high)
        | (metrics["length_disp_ratio"] >= ratio_high)
    ].copy()
    outliers.to_csv(outdir / "flight_outliers.csv", index=False)

    return {
        "length_q05": float(q_low),
        "length_q95": float(q_high),
        "ratio_q95": float(ratio_high),
        "outliers_count": int(len(outliers)),
    }


def _plot_hist(series: pd.Series, outpath: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(series.dropna(), bins=50, color="#4C78A8", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _plot_scatter(x: pd.Series, y: pd.Series, outpath: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=8, alpha=0.25, color="#F58518")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _plot_density(df: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.hist2d(df["x_utm"], df["y_utm"], bins=150, cmap="viridis")
    plt.title("UTM density heatmap")
    plt.xlabel("x_utm")
    plt.ylabel("y_utm")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _plot_sample_trajectories(df: pd.DataFrame, metrics: pd.DataFrame, outpath: Path, config: EdaConfig) -> None:
    rng = np.random.default_rng(config.seed)
    sample_ids: List[str] = []

    if config.sample_per_flow > 0:
        for (_, _), group in metrics.groupby(["A/D", "Runway"]):
            ids = group["flight_id"].to_numpy()
            if ids.size == 0:
                continue
            take = min(config.sample_per_flow, ids.size)
            sample_ids.extend(list(rng.choice(ids, size=take, replace=False)))

    if config.sample_total > 0:
        remaining = max(0, config.sample_total - len(sample_ids))
        if remaining > 0:
            ids = metrics["flight_id"].to_numpy()
            extra = rng.choice(ids, size=min(remaining, ids.size), replace=False)
            sample_ids.extend(list(extra))

    sample_ids = list(dict.fromkeys(sample_ids))
    if not sample_ids:
        return

    df_sample = df[df["flight_id"].isin(sample_ids)].copy()
    df_sample = df_sample.sort_values(["flight_id", "step"])
    color_map = {
        ("Landung", "09L"): "#1f77b4",
        ("Landung", "09R"): "#ff7f0e",
        ("Landung", "27L"): "#2ca02c",
        ("Landung", "27R"): "#d62728",
        ("Start", "09L"): "#9467bd",
        ("Start", "09R"): "#8c564b",
        ("Start", "27L"): "#e377c2",
        ("Start", "27R"): "#7f7f7f",
    }

    plt.figure(figsize=(7, 7))
    for flight_id, group in df_sample.groupby("flight_id"):
        ad = group["A/D"].iloc[0]
        rw = group["Runway"].iloc[0]
        color = color_map.get((ad, rw), "#17becf")
        plt.plot(group["x_utm"], group["y_utm"], color=color, alpha=0.5, linewidth=0.7)

    plt.title("Sample trajectories (UTM)")
    plt.xlabel("x_utm")
    plt.ylabel("y_utm")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def run_eda(config: EdaConfig) -> None:
    config.outdir.mkdir(parents=True, exist_ok=True)

    df = _read_csv(config.input_csv, max_rows=config.max_rows)
    metrics = _compute_flight_metrics(df)

    metrics.to_csv(config.outdir / "flight_metrics.csv", index=False)

    summary = _write_counts(metrics, config.outdir)
    summary.update(_write_outliers(metrics, config.outdir))

    # Plots
    _plot_hist(metrics["length_m"], config.outdir / "length_hist.png", "Flight length (m)", "length_m")
    _plot_hist(metrics["displacement_m"], config.outdir / "displacement_hist.png", "Displacement (m)", "displacement_m")
    _plot_hist(metrics["length_disp_ratio"], config.outdir / "length_disp_ratio_hist.png", "Length/Displacement", "ratio")
    _plot_scatter(
        metrics["length_m"],
        metrics["displacement_m"],
        config.outdir / "length_vs_displacement.png",
        "Length vs displacement",
        "length_m",
        "displacement_m",
    )
    _plot_density(df, config.outdir / "utm_density.png")
    _plot_sample_trajectories(df, metrics, config.outdir / "sample_trajectories.png", config)

    summary_path = config.outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("EDA completed")
    print(f"Summary: {summary_path}")
    print(f"Counts by flow: {config.outdir / 'counts_by_flow.csv'}")
    print(f"Flight metrics: {config.outdir / 'flight_metrics.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for UTM trajectory CSVs.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--sample-total", type=int, default=200)
    parser.add_argument("--sample-per-flow", type=int, default=25)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    logs_dir = Path("logs") / "eda"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "eda_utm.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    config = EdaConfig(
        input_csv=Path(args.input),
        outdir=Path(args.outdir),
        sample_total=args.sample_total,
        sample_per_flow=args.sample_per_flow,
        seed=args.seed,
        max_rows=args.max_rows,
    )

    logging.info("Running EDA for %s", config.input_csv)
    run_eda(config)


if __name__ == "__main__":
    main()
