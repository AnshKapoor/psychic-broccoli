"""Visualize backbone tracks produced by generate_backbone_tracks.py.

This script renders per-flow centrelines (lat/lon) and vertical/energy
profiles (altitude, groundspeed) using the p10/p50/p90 envelopes from
``backbone_tracks.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_NUMERIC = ("latitude", "longitude", "altitude", "groundspeed")


def parse_flow(flow: str) -> Tuple[str, str]:
    """Split a flow string of the form 'A/D:Runway' into its parts."""

    if ":" not in flow:
        raise ValueError("Flow must be formatted as 'A/D:Runway', e.g., 'Start:27R' or 'Landung:09L'.")
    ad, runway = flow.split(":", 1)
    return ad.strip(), runway.strip()


def filter_flows(df: pd.DataFrame, flows: Iterable[str]) -> pd.DataFrame:
    """Return only the requested flows."""

    pairs = [parse_flow(f) for f in flows]
    mask = False
    for ad, rw in pairs:
        mask = mask | ((df["A/D"] == ad) & (df["Runway"] == rw))
    return df[mask]


def plot_flow(df: pd.DataFrame, flow_label: str, save_path: Path | None) -> None:
    """Plot a single flow with lat/lon path and envelopes for altitude/groundspeed."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    lat_cols = ("latitude_p10", "latitude_p50", "latitude_p90")
    lon_cols = ("longitude_p10", "longitude_p50", "longitude_p90")

    axes[0].plot(df[lon_cols[1]], df[lat_cols[1]], label="p50 centreline", color="tab:blue")
    axes[0].fill_between(df[lon_cols[1]], df[lat_cols[0]], df[lat_cols[2]], color="tab:blue", alpha=0.2, label="p10–p90")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title(f"Ground track ({flow_label})")
    axes[0].legend()

    axes[1].plot(df["step"], df["altitude_p50"], color="tab:green", label="p50")
    axes[1].fill_between(df["step"], df["altitude_p10"], df["altitude_p90"], color="tab:green", alpha=0.2, label="p10–p90")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Altitude")
    axes[1].set_title("Altitude envelope")
    axes[1].legend()

    axes[2].plot(df["step"], df["groundspeed_p50"], color="tab:orange", label="p50")
    axes[2].fill_between(df["step"], df["groundspeed_p10"], df["groundspeed_p90"], color="tab:orange", alpha=0.2, label="p10–p90")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Groundspeed")
    axes[2].set_title("Groundspeed envelope")
    axes[2].legend()

    fig.suptitle(f"Backbone flow: {flow_label} (n={int(df['n_flights'].iloc[0])})", fontsize=12)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize backbone tracks from backbone_tracks.csv")
    parser.add_argument("--csv", type=Path, default=Path("reports/backbone_tracks.csv"))
    parser.add_argument(
        "--flow",
        action="append",
        help="Flow to plot formatted as 'A/D:Runway', e.g., 'Start:27R'. Can be passed multiple times.",
    )
    parser.add_argument("--save-dir", type=Path, help="If set, save PNGs to this directory instead of showing them.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required_cols = ["A/D", "Runway", "step"] + [f"{c}_p50" for c in DEFAULT_NUMERIC]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}")

    if args.flow:
        df = filter_flows(df, args.flow)
        if df.empty:
            raise ValueError("No rows matched the requested flow(s).")

    # Plot each flow independently.
    for (ad, runway), group in df.groupby(["A/D", "Runway"]):
        flow_label = f"{ad}:{runway}"
        save_path = args.save_dir / f"backbone_{ad}_{runway}.png" if args.save_dir else None
        plot_flow(group.sort_values("step"), flow_label, save_path)


if __name__ == "__main__":
    main()
