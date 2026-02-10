"""Aggregate cluster-based Doc29 outputs and compare to ground truth.

Usage:
  python noise_simulation/aggregate_cluster_cumulative.py \
    --summary noise_simulation/results/EXP001/summary_mse.csv \
    --ground-truth noise_simulation/results_ground_truth/preprocessed_1/ground_truth_cumulative.csv \
    --out noise_simulation/results/EXP001/aggregate

Inputs:
  - summary_mse.csv with columns:
      experiment,A/D,Runway,cluster_id,aircraft_type,n_flights,subtracks_csv,groundtruth_csv
  - ground_truth_cumulative.csv (semicolon CSV) with columns:
      x;y;z;...;cumulative_res

Outputs:
  - cluster_prediction_cumulative.csv (semicolon CSV with x,y,z,cumulative_res)
  - ground_truth_cumulative_aligned.csv (same compared rows/shape as prediction)
  - mae_summary.json (MAE/MSE/RMSE on cumulative_res)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _to_energy(level_db: np.ndarray) -> np.ndarray:
    return np.power(10.0, level_db / 10.0)


def _to_db(energy: np.ndarray) -> np.ndarray:
    energy = np.maximum(energy, 1e-12)
    return 10.0 * np.log10(energy)


def _add_energy(
    accum: Optional[pd.DataFrame],
    new: pd.DataFrame,
    scale: float,
) -> pd.DataFrame:
    """Sum cumulative_res in energy domain, scaled by flight count factor."""

    cols = ["x", "y", "z", "cumulative_res"]
    if not set(cols).issubset(new.columns):
        raise ValueError(f"Missing required columns in Doc29 output: {cols}")
    new = new[cols].copy()
    new["energy"] = _to_energy(new["cumulative_res"].to_numpy()) * float(scale)
    new = new.drop(columns=["cumulative_res"])

    if accum is None:
        return new
    merged = accum.merge(new, on=["x", "y", "z"], how="outer", suffixes=("_a", "_b"))
    merged["energy"] = merged["energy_a"].fillna(0.0) + merged["energy_b"].fillna(0.0)
    return merged[["x", "y", "z", "energy"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate cluster subtracks and compare to ground truth.")
    parser.add_argument("--summary", required=True, help="Path to summary_mse.csv (from run_doc29_experiment.py).")
    parser.add_argument("--ground-truth", required=True, help="Path to ground_truth_cumulative.csv.")
    parser.add_argument("--out", required=True, help="Output folder for aggregate results.")
    parser.add_argument("--tracks-per-cluster", type=int, default=7, help="Number of subtracks per cluster.")
    parser.add_argument(
        "--subtracks-weighting",
        choices=["weighted", "unweighted"],
        default="weighted",
        help=(
            "How Flight_subtracks.csv was configured. "
            "'weighted' means Nr.day already contains per-track flight counts (no extra scaling). "
            "'unweighted' means each subtrack represents one virtual flight (apply n_flights/tracks_per_cluster scaling)."
        ),
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    gt_path = Path(args.ground_truth)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path)
    required = {"n_flights", "subtracks_csv"}
    if not required.issubset(summary.columns):
        raise ValueError(f"summary_mse.csv missing columns: {sorted(required - set(summary.columns))}")

    accum: Optional[pd.DataFrame] = None
    skipped = 0
    for _, row in summary.iterrows():
        sub_path = Path(str(row["subtracks_csv"]))
        if not sub_path.exists():
            skipped += 1
            continue
        n_flights = int(row["n_flights"])
        if args.subtracks_weighting == "weighted":
            scale = 1.0
        else:
            scale = n_flights / float(args.tracks_per_cluster)
        sub_df = pd.read_csv(sub_path, sep=";")
        accum = _add_energy(accum, sub_df, scale=scale)

    if accum is None:
        raise RuntimeError("No subtracks could be aggregated. Check summary_mse.csv paths.")

    accum["cumulative_res"] = _to_db(accum["energy"].to_numpy())
    pred = accum[["x", "y", "z", "cumulative_res"]]
    pred_path = out_dir / "cluster_prediction_cumulative.csv"
    pred.to_csv(pred_path, sep=";", index=False)

    gt = pd.read_csv(gt_path, sep=";")
    join_cols = [col for col in ("x", "y", "z") if col in gt.columns and col in pred.columns]
    if join_cols:
        merged = pred.merge(gt, on=join_cols, suffixes=("_pred", "_gt"))
    else:
        merged = pd.DataFrame(
            {"cumulative_res_pred": pred["cumulative_res"], "cumulative_res_gt": gt["cumulative_res"]}
        )
    if merged.empty:
        raise ValueError("No overlapping rows to compare between prediction and ground truth.")

    # Persist aligned views used in final comparison.
    if join_cols:
        pred_aligned = merged[join_cols + ["cumulative_res_pred"]].rename(
            columns={"cumulative_res_pred": "cumulative_res"}
        )
        gt_aligned = merged[join_cols + ["cumulative_res_gt"]].rename(
            columns={"cumulative_res_gt": "cumulative_res"}
        )
    else:
        pred_aligned = pd.DataFrame({"cumulative_res": merged["cumulative_res_pred"]})
        gt_aligned = pd.DataFrame({"cumulative_res": merged["cumulative_res_gt"]})

    pred_aligned_path = out_dir / "cluster_prediction_cumulative_aligned.csv"
    gt_aligned_path = out_dir / "ground_truth_cumulative_aligned.csv"
    pred_aligned.to_csv(pred_aligned_path, sep=";", index=False)
    gt_aligned.to_csv(gt_aligned_path, sep=";", index=False)

    diff = merged["cumulative_res_pred"] - merged["cumulative_res_gt"]
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(np.square(diff)))
    rmse = float(np.sqrt(mse))
    summary_out = {
        "summary_mse": str(summary_path),
        "ground_truth": str(gt_path),
        "prediction": str(pred_path),
        "prediction_aligned": str(pred_aligned_path),
        "ground_truth_aligned": str(gt_aligned_path),
        "tracks_per_cluster": args.tracks_per_cluster,
        "subtracks_weighting": args.subtracks_weighting,
        "rows_compared": int(len(merged)),
        "skipped_subtracks": int(skipped),
        "mae_cumulative_res": mae,
        "mse_cumulative_res": mse,
        "rmse_cumulative_res": rmse,
    }
    (out_dir / "mae_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    print(f"Wrote {pred_path}")
    print(f"Wrote {out_dir / 'mae_summary.json'}")


if __name__ == "__main__":
    main()
