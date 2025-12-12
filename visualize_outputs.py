"""Quick visualisations for backbone, clustered, and preprocessed outputs.

Saves PNGs under output/run/figures. By default it uses the latest CSVs in
output/run/csv. You can override which files are selected with
--experiment-prefix, which matches the common part in filenames
(e.g., 'OPTICS_exp_3' to pick preprocessed_OPTICS_exp_3.csv, etc.).

Generated plots:
- backbones_*: median tracks with p10–p90 band per flow
- clustered_flights_*: trajectories per cluster (noise removed)
- preprocessed_sample_*: sample trajectories (up to 50)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import argparse

BASE_RUN_DIR = Path("output") / "run"
CSV_DIR = BASE_RUN_DIR / "csv"
FIG_DIR = BASE_RUN_DIR / "figures"


def _find_latest_file(prefix: str, suffix: str = ".csv", experiment_prefix: str | None = None) -> Path | None:
    if not CSV_DIR.exists():
        return None
    pattern = f"{prefix}_*{experiment_prefix}*{suffix}" if experiment_prefix else f"{prefix}_*{suffix}"
    candidates = sorted(CSV_DIR.glob(pattern))
    return candidates[-1] if candidates else None


def plot_backbones(path: Path, plots_dir: Path) -> None:
    df = pd.read_csv(path) if path and path.exists() else pd.DataFrame()
    if df.empty:
        print(f"{path} is empty or missing.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    color_cycle = plt.cm.get_cmap("tab20", max(1, df["Runway"].nunique() * 2))
    color_idx = 0
    has_percentile = "percentile" in df.columns
    for (ad, rw), group in df.groupby(["A/D", "Runway"]):
        color = color_cycle(color_idx % color_cycle.N)
        color_idx += 1
        if has_percentile:
            p50 = group[group["percentile"] == 50]
            p10 = group[group["percentile"] == 10]
            p90 = group[group["percentile"] == 90]
            if not p10.empty:
                ax.plot(p10["longitude"], p10["latitude"], label=f"{ad}-{rw} p10", lw=1.5, color=color)
            if not p90.empty:
                ax.plot(p90["longitude"], p90["latitude"], label=f"{ad}-{rw} p90", lw=1.5, color=color)
            if not p50.empty:
                ax.plot(p50["longitude"], p50["latitude"], label=f"{ad}-{rw} p50", lw=2, alpha=0.5, color=color)
        else:
            ax.plot(group["longitude"], group["latitude"], label=f"{ad}-{rw}", lw=2, color=color)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.set_title("Backbone tracks")
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"backbones_{path.stem.replace('.csv','')}.png", dpi=200)
    plt.close(fig)


def plot_clustered(path: Path, plots_dir: Path, ad: str | None = None, runway: str | None = None) -> None:
    df = pd.read_csv(path) if path and path.exists() else pd.DataFrame()
    if df.empty:
        print(f"{path} is empty or missing.")
        return

    if ad:
        df = df[df["A/D"] == ad]
    if runway:
        df = df[df["Runway"] == runway]

    noise_count = int((df["cluster_id"] == -1).sum()) if "cluster_id" in df.columns else 0
    df = df[df["cluster_id"] != -1]
    if df.empty:
        print("No clustered points after filtering/noise removal.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    clusters = sorted(df["cluster_id"].unique())
    cmap = plt.cm.get_cmap("tab10", max(len(clusters), 1))
    for cid in clusters:
        subset = df[df["cluster_id"] == cid]
        for _, flight in subset.groupby("flight_id"):
            ax.plot(flight["longitude"], flight["latitude"], color=cmap(cid), alpha=0.3, lw=0.8)
    sc = ax.scatter([], [], c=[], cmap="tab10")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title_flow = f"{ad or 'all'}/{runway or 'all'}"
    ax.set_title(f"Clustered flights (trajectories, {title_flow})")
    fig.colorbar(sc, ax=ax, label="cluster_id", ticks=clusters, boundaries=[min(clusters)-0.5, max(clusters)+0.5])
    ax.text(0.01, 0.99, f"Noise dropped: {noise_count}", transform=ax.transAxes, va="top", ha="left", fontsize=8)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{ad}_{runway}".replace("/", "_") if ad or runway else ""
    fig.savefig(plots_dir / f"clustered_flights{suffix}_{path.stem.replace('.csv','')}.png", dpi=200)
    plt.close(fig)


def plot_preprocessed(path: Path, plots_dir: Path, max_trajs: int = 200) -> None:
    df = pd.read_csv(path) if path and path.exists() else pd.DataFrame()
    if df.empty:
        print(f"{path} is empty or missing.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for (_, _, _), flight in df.groupby(["A/D", "Runway", "flight_id"]):
        ax.plot(flight["longitude"], flight["latitude"], alpha=0.2, lw=1)
        max_trajs -= 1
        if max_trajs <= 0:
            break
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Sample of preprocessed trajectories (up to 50)")
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"preprocessed_sample_{path.stem.replace('.csv','')}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize backbone/clustering outputs.")
    parser.add_argument(
        "--experiment-prefix",
        default=None,
        help="Common experiment part in filenames (e.g., OPTICS_exp_3). Uses latest files if omitted.",
    )
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    backbone_path = _find_latest_file("backbone_tracks", experiment_prefix=args.experiment_prefix)
    clustered_path = _find_latest_file("clustered_flights", experiment_prefix=args.experiment_prefix)
    preprocessed_path = _find_latest_file("preprocessed", experiment_prefix=args.experiment_prefix)

    plot_backbones(backbone_path, FIG_DIR)
    plot_clustered(clustered_path, FIG_DIR)
    # Example per-flow plots; uncomment if desired:
    # plot_clustered(clustered_path, FIG_DIR, ad="Landung", runway="27R")
    # plot_clustered(clustered_path, FIG_DIR, ad="Start", runway="09L")
    plot_preprocessed(preprocessed_path, FIG_DIR)
    print(f"Saved plots to {FIG_DIR}")
