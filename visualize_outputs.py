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
            ax.plot(p50["longitude"], p50["latitude"], label=f"{ad}-{rw} p50", lw=2)
            if not p10.empty and not p90.empty:
                # Shade between p10 and p90 along step order
                merged = p50[["step", "longitude"]].merge(
                    p10[["step", "latitude"]].rename(columns={"latitude": "lat_p10"}), on="step", how="left"
                ).merge(
                    p90[["step", "latitude"]].rename(columns={"latitude": "lat_p90"}), on="step", how="left"
                )
                ax.fill_between(
                    merged["longitude"],
                    merged["lat_p10"],
                    merged["lat_p90"],
                    alpha=0.2,
                    label=f"{ad}-{rw} p10-p90",
                )
        else:
            ax.plot(group["longitude"], group["latitude"], label=f"{ad}-{rw}", lw=2)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.set_title("Backbone tracks")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "backbones.png", dpi=200)
    plt.close(fig)


def plot_clustered(path: Path = OUT_DIR / "clustered_flights.csv") -> None:
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if df.empty:
        print(f"{path} is empty or missing.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    last_pts = df.sort_values("step").groupby("flight_id").tail(1)
    sc = ax.scatter(last_pts["longitude"], last_pts["latitude"], c=last_pts["cluster_id"], cmap="tab10", s=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Clustered flights (last point per flight, coloured by cluster_id)")
    fig.colorbar(sc, ax=ax, label="cluster_id")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "clustered_flights.png", dpi=200)
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
    plot_preprocessed()
    print(f"Saved plots to {PLOTS_DIR}")
