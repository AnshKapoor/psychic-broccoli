"""Update metrics_global.csv with NaN-excluded weighted metrics for all experiments.

Usage:
  python scripts/fix_metrics_global.py --experiments-dir output/experiments

Behavior:
  - Reads metrics_by_flow.csv for each EXP directory.
  - Computes weighted averages over flows with valid metrics (>=2 clusters).
  - Appends new columns to metrics_global.csv:
      * silhouette_valid
      * davies_bouldin_valid
      * calinski_harabasz_valid
      * n_flows_total
      * n_flows_valid
  - If metrics_global.csv is missing, creates it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    vals = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    return float(np.average(vals, weights=w))


def compute_valid_metrics(df: pd.DataFrame) -> dict:
    metrics = {}
    total_flows = len(df)
    metrics["n_flows_total"] = int(total_flows)
    valid_mask = df[["silhouette", "davies_bouldin", "calinski_harabasz"]].notna().all(axis=1)
    metrics["n_flows_valid"] = int(valid_mask.sum())

    if valid_mask.any():
        weights = df.loc[valid_mask, "n_flights"]
        metrics["silhouette_valid"] = weighted_mean(df.loc[valid_mask, "silhouette"], weights)
        metrics["davies_bouldin_valid"] = weighted_mean(df.loc[valid_mask, "davies_bouldin"], weights)
        metrics["calinski_harabasz_valid"] = weighted_mean(df.loc[valid_mask, "calinski_harabasz"], weights)
    else:
        metrics["silhouette_valid"] = np.nan
        metrics["davies_bouldin_valid"] = np.nan
        metrics["calinski_harabasz_valid"] = np.nan

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix metrics_global.csv by excluding NaN flows.")
    parser.add_argument("--experiments-dir", default="output/experiments")
    args = parser.parse_args()

    exp_root = Path(args.experiments_dir)
    if not exp_root.exists():
        raise FileNotFoundError(f"Experiments dir not found: {exp_root}")

    for exp_dir in sorted(p for p in exp_root.iterdir() if p.is_dir() and p.name.startswith("EXP")):
        by_flow = exp_dir / "metrics_by_flow.csv"
        if not by_flow.exists():
            continue

        df = pd.read_csv(by_flow)
        if df.empty:
            continue

        valid_metrics = compute_valid_metrics(df)
        global_path = exp_dir / "metrics_global.csv"

        if global_path.exists():
            gdf = pd.read_csv(global_path)
            if gdf.empty:
                gdf = pd.DataFrame([{}])
        else:
            gdf = pd.DataFrame([{}])

        for k, v in valid_metrics.items():
            gdf.loc[0, k] = v

        gdf.to_csv(global_path, index=False)

    print(f"Updated metrics_global.csv in {exp_root}")


if __name__ == "__main__":
    main()

