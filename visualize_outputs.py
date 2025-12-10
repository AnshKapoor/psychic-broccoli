"""Quick visualisations for backbone, clustered, and preprocessed outputs.

Generates three PNGs under output/plots:
- backbones.png: median tracks with p10–p90 band per flow
- clustered_flights.png: last point of each flight coloured by cluster_id
- preprocessed_sample.png: sample trajectories (up to 50)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("output")
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_backbones(path: Path = OUT_DIR / "backbone_tracks.csv") -> None:
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if df.empty:
        print(f"{path} is empty or missing.")
        return

    # Handle percentile+long format: expect columns percentile, longitude/latitude
    # If percentile bands are absent, just plot the points.
    fig, ax = plt.subplots(figsize=(8, 6))
    has_percentile = "percentile" in df.columns
    for (ad, rw), group in df.groupby(["A/D", "Runway"]):
        if has_percentile:
            p50 = group[group["percentile"] == 50]
            p10 = group[group["percentile"] == 10]
            p90 = group[group["percentile"] == 90]
            # Plot p10/p90 as solid lines; p50 as a lighter line of the same color.
            if not p10.empty:
                ax.plot(p10["longitude"], p10["latitude"], label=f"{ad}-{rw} p10", lw=1.5)
            if not p90.empty:
                ax.plot(p90["longitude"], p90["latitude"], label=f"{ad}-{rw} p90", lw=1.5)
            if not p50.empty:
                ax.plot(p50["longitude"], p50["latitude"], label=f"{ad}-{rw} p50", lw=2, alpha=0.5)
        else:
            ax.plot(group["longitude"], group["latitude"], label=f"{ad}-{rw}", lw=2)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.set_title("Backbone tracks")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "backbones.png", dpi=200)
    plt.close(fig)


def plot_clustered(path: Path = OUT_DIR / "clustered_flights.csv", ad: str | None = None, runway: str | None = None) -> None:
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if df.empty:
        print(f"{path} is empty or missing.")
        return

    if ad:
        df = df[df["A/D"] == ad]
    if runway:
        df = df[df["Runway"] == runway]

    # Drop noise for clarity
    df = df[df["cluster_id"] != -1]
    if df.empty:
        print("No clustered points after filtering/noise removal.")
        return

    # Plot full trajectories per cluster
    fig, ax = plt.subplots(figsize=(8, 6))
    clusters = sorted(df["cluster_id"].unique())
    cmap = plt.cm.get_cmap("tab10", max(len(clusters), 1))
    for cid in clusters:
        subset = df[df["cluster_id"] == cid]
        for _, flight in subset.groupby("flight_id"):
            ax.plot(flight["longitude"], flight["latitude"], color=cmap(cid), alpha=0.3, lw=0.8)
    sc = ax.scatter([], [], c=[], cmap="tab10")  # placeholder for consistent colorbar
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title_flow = f"{ad or 'all'}/{runway or 'all'}"
    ax.set_title(f"Clustered flights (trajectories, {title_flow})")
    fig.colorbar(sc, ax=ax, label="cluster_id", ticks=clusters, boundaries=[min(clusters)-0.5, max(clusters)+0.5])
    fig.tight_layout()
    suffix = f"_{ad}_{runway}".replace("/", "_") if ad or runway else ""
    fig.savefig(PLOTS_DIR / f"clustered_flights{suffix}.png", dpi=200)
    plt.close(fig)


def plot_preprocessed(path: Path = OUT_DIR / "preprocessed.csv", max_trajs: int = 50) -> None:
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
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
    fig.savefig(PLOTS_DIR / "preprocessed_sample.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_backbones()
    plot_clustered()
    # Example per-flow plots; uncomment if desired:
    # plot_clustered(ad="Landung", runway="27R")
    # plot_clustered(ad="Start", runway="09L")
    plot_preprocessed()
    print(f"Saved plots to {PLOTS_DIR}")
