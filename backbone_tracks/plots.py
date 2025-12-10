"""Optional plotting utilities for debugging.

Provides simple matplotlib helpers to visualise segmentation and backbone
envelopes per flow/cluster, supporting UTM or lat/lon views.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_flight_segmentation(df: pd.DataFrame, ad: str, runway: str, output_path: Path) -> None:
    """Plot distance-to-airport over time, colored by flight_id."""

    subset = df[(df["A/D"] == ad) & (df["Runway"] == runway)]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for flight_id, group in subset.groupby("flight_id"):
        ax.plot(group["timestamp"], group["dist_to_airport_m"], label=f"flight {flight_id}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Distance to airport (m)")
    ax.set_title(f"Segmentation {ad} {runway}")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_backbone(
    df: pd.DataFrame,
    ad: str,
    runway: str,
    cluster_id: int,
    output_path: Path,
    use_utm: bool = False,
) -> None:
    """Plot backbone percentiles for a given cluster."""

    subset = df[(df["A/D"] == ad) & (df["Runway"] == runway) & (df["cluster_id"] == cluster_id)]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    for p, style in [(50, "-"), (10, "--"), (90, "--")]:
        band = subset[subset["percentile"] == p].sort_values("step")
        if use_utm and "x_utm" in band and "y_utm" in band:
            ax.plot(band["x_utm"], band["y_utm"], style, label=f"p{p}")
            ax.set_xlabel("X UTM (m)")
            ax.set_ylabel("Y UTM (m)")
        else:
            ax.plot(band["longitude"], band["latitude"], style, label=f"p{p}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
    ax.set_title(f"Backbone {ad} {runway} cluster {cluster_id}")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
