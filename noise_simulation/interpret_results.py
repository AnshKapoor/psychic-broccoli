"""Summarize Doc29 MSE results into an interpretation folder.

Usage:
  python noise_simulation/interpret_results.py --summary noise_simulation/results/EXP46.../summary_mse.csv

Example input format (summary_mse.csv):
  experiment,A/D,Runway,cluster_id,aircraft_type,n_flights,mse_cumulative_res,n_measurements,subtracks_csv,groundtruth_csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    return float((values * weights).sum() / total)


def _flow_summary(df: pd.DataFrame) -> pd.DataFrame:
    def summarize(group: pd.DataFrame) -> pd.Series:
        total_flights = group["n_flights"].sum()
        return pd.Series(
            {
                "n_rows": len(group),
                "n_clusters": group["cluster_id"].nunique(),
                "n_aircraft_types": group["aircraft_type"].nunique(),
                "total_flights": total_flights,
                "mse_mean": group["mse_cumulative_res"].mean(),
                "mse_median": group["mse_cumulative_res"].median(),
                "mse_min": group["mse_cumulative_res"].min(),
                "mse_max": group["mse_cumulative_res"].max(),
                "mse_weighted": _weighted_mean(group["mse_cumulative_res"], group["n_flights"]),
            }
        )

    return (
        df.groupby(["experiment", "A/D", "Runway"], as_index=False)
        .apply(summarize)
        .reset_index(drop=True)
    )


def _flow_aircraft_summary(df: pd.DataFrame) -> pd.DataFrame:
    def summarize(group: pd.DataFrame) -> pd.Series:
        total_flights = group["n_flights"].sum()
        return pd.Series(
            {
                "n_rows": len(group),
                "n_clusters": group["cluster_id"].nunique(),
                "total_flights": total_flights,
                "mse_mean": group["mse_cumulative_res"].mean(),
                "mse_median": group["mse_cumulative_res"].median(),
                "mse_min": group["mse_cumulative_res"].min(),
                "mse_max": group["mse_cumulative_res"].max(),
                "mse_weighted": _weighted_mean(group["mse_cumulative_res"], group["n_flights"]),
            }
        )

    return (
        df.groupby(["experiment", "A/D", "Runway", "aircraft_type"], as_index=False)
        .apply(summarize)
        .reset_index(drop=True)
    )


def _type_delta(flow_type_summary: pd.DataFrame, type_a: str, type_b: str) -> pd.DataFrame:
    keys = ["experiment", "A/D", "Runway"]
    a_df = flow_type_summary[flow_type_summary["aircraft_type"] == type_a].copy()
    b_df = flow_type_summary[flow_type_summary["aircraft_type"] == type_b].copy()

    a_df = a_df.rename(
        columns={
            "total_flights": f"total_flights_{type_a}",
            "n_clusters": f"n_clusters_{type_a}",
            "mse_mean": f"mse_mean_{type_a}",
            "mse_weighted": f"mse_weighted_{type_a}",
        }
    )
    b_df = b_df.rename(
        columns={
            "total_flights": f"total_flights_{type_b}",
            "n_clusters": f"n_clusters_{type_b}",
            "mse_mean": f"mse_mean_{type_b}",
            "mse_weighted": f"mse_weighted_{type_b}",
        }
    )

    merged = pd.merge(a_df[keys + [f"total_flights_{type_a}", f"n_clusters_{type_a}", f"mse_mean_{type_a}", f"mse_weighted_{type_a}"]],
                      b_df[keys + [f"total_flights_{type_b}", f"n_clusters_{type_b}", f"mse_mean_{type_b}", f"mse_weighted_{type_b}"]],
                      on=keys,
                      how="outer")

    merged["delta_mse_mean"] = merged.get(f"mse_mean_{type_b}") - merged.get(f"mse_mean_{type_a}")
    merged["delta_mse_weighted"] = merged.get(f"mse_weighted_{type_b}") - merged.get(f"mse_weighted_{type_a}")
    return merged


def _threshold_report(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return df[df["mse_cumulative_res"] >= threshold].sort_values("mse_cumulative_res", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Doc29 MSE results into an interpretation folder.")
    parser.add_argument("--summary", required=True, help="Path to summary_mse.csv.")
    parser.add_argument("--out-dir", default=None, help="Output folder (defaults to <summary_dir>/interpretation).")
    parser.add_argument("--threshold", type=float, default=5.0, help="MSE threshold for the high-error report.")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    df = pd.read_csv(summary_path)
    required = {
        "experiment",
        "A/D",
        "Runway",
        "cluster_id",
        "aircraft_type",
        "n_flights",
        "mse_cumulative_res",
    }
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(f"summary_mse.csv missing columns: {sorted(missing)}")

    out_dir = Path(args.out_dir) if args.out_dir else (summary_path.parent / "interpretation")
    out_dir.mkdir(parents=True, exist_ok=True)

    flow_summary = _flow_summary(df)
    flow_type_summary = _flow_aircraft_summary(df)
    delta_df = _type_delta(flow_type_summary, "A320", "A321")
    over_df = _threshold_report(df, args.threshold)

    flow_summary.to_csv(out_dir / "flow_summary.csv", index=False)
    flow_type_summary.to_csv(out_dir / "flow_aircraft_type_summary.csv", index=False)
    delta_df.to_csv(out_dir / "flow_type_delta_A320_A321.csv", index=False)
    over_df.to_csv(out_dir / f"mse_over_{args.threshold:g}.csv", index=False)

    meta: Dict[str, object] = {
        "summary_source": str(summary_path),
        "threshold": args.threshold,
        "rows": len(df),
        "flows": int(df[["A/D", "Runway"]].drop_duplicates().shape[0]),
        "aircraft_types": int(df["aircraft_type"].nunique()),
    }
    (out_dir / "interpretation_meta.json").write_text(
        pd.Series(meta).to_json(indent=2),
        encoding="utf-8",
    )

    print(f"Wrote interpretation files to {out_dir}")


if __name__ == "__main__":
    main()
