"""Plot clustering metrics for Doc29 experiments.

Reads the aggregated metrics CSVs and produces simple bar plots to compare
silhouette, Davies-Bouldin, and Calinski-Harabasz scores grouped by experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


def _load_csv(path: Path, required_cols: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def _plot_grouped(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    hue_col: str | None = None,
) -> None:
    plt.figure(figsize=(10, 5))
    if hue_col and hue_col in df.columns:
        for key, sub in df.groupby(hue_col):
            plt.bar(sub[x_col], sub[y_col], label=str(key), alpha=0.7)
        plt.legend(title=hue_col)
    else:
        plt.bar(df[x_col], df[y_col], alpha=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(y_col)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Doc29 experiment metrics.")
    parser.add_argument(
        "--stage2",
        default="output/eda/metrics_stage2_by_flow.csv",
        help="Path to metrics_stage2_exp32_37_by_flow.csv",
    )
    parser.add_argument(
        "--quality",
        default="output/eda/metrics_quality_global.csv",
        help="Path to metrics_quality_exp01_31.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="output/eda/figures",
        help="Directory to write plots.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    stage2 = _load_csv(
        Path(args.stage2),
        required_cols=["experiment_name", "silhouette", "davies_bouldin", "calinski_harabasz"],
    )
    quality = _load_csv(
        Path(args.quality),
        required_cols=["experiment_name", "silhouette", "davies_bouldin", "calinski_harabasz"],
    )

    # Stage 2: drop flows without clusters
    stage2_valid = stage2[stage2["silhouette"].notna() & stage2["experiment_name"].notna()].copy()
    stage2_valid["exp_short"] = stage2_valid["experiment_name"].astype(str).str.replace("EXP", "E", regex=False)

    # Quality set
    quality_valid = quality[quality["silhouette"].notna()].copy()

    _plot_grouped(
        stage2_valid,
        x_col="exp_short",
        y_col="silhouette",
        title="Stage2 Silhouette by Experiment (per flow)",
        out_path=out_dir / "stage2_silhouette.png",
        hue_col="A/D" if "A/D" in stage2_valid.columns else None,
    )
    _plot_grouped(
        stage2_valid,
        x_col="exp_short",
        y_col="davies_bouldin",
        title="Stage2 Davies-Bouldin (per flow)",
        out_path=out_dir / "stage2_davies_bouldin.png",
        hue_col="A/D" if "A/D" in stage2_valid.columns else None,
    )
    _plot_grouped(
        stage2_valid,
        x_col="exp_short",
        y_col="calinski_harabasz",
        title="Stage2 Calinski-Harabasz (per flow)",
        out_path=out_dir / "stage2_calinski.png",
        hue_col="A/D" if "A/D" in stage2_valid.columns else None,
    )

    _plot_grouped(
        quality_valid,
        x_col="experiment_name",
        y_col="silhouette",
        title="EXP01–31 Silhouette",
        out_path=out_dir / "quality_silhouette.png",
    )
    _plot_grouped(
        quality_valid,
        x_col="experiment_name",
        y_col="davies_bouldin",
        title="EXP01–31 Davies-Bouldin",
        out_path=out_dir / "quality_davies_bouldin.png",
    )
    _plot_grouped(
        quality_valid,
        x_col="experiment_name",
        y_col="calinski_harabasz",
        title="EXP01–31 Calinski-Harabasz",
        out_path=out_dir / "quality_calinski.png",
    )

    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
