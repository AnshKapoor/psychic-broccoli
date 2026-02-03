"""Plot clustering results from an experiment output folder.

Usage:
  python scripts/plot_experiment_results.py OPTICS_exp_5

This reads outputs from `output/experiments/<experiment_name>/`:
  - metrics_by_flow.csv / metrics_global.csv
  - labels_*.csv (one per flow)
  - config_resolved.yaml (to locate preprocessed CSV)

It writes plots to:
  output/experiments/<experiment_name>/plots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone_tracks.backbone import compute_backbones


def _load_resolved_config(exp_dir: Path) -> dict:
    cfg_path = exp_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _ensure_plots_dir(exp_dir: Path) -> Path:
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def _plot_metrics(exp_dir: Path, plots_dir: Path) -> None:
    by_flow = exp_dir / "metrics_by_flow.csv"
    global_path = exp_dir / "metrics_global.csv"
    if not by_flow.exists():
        return

    df = pd.read_csv(by_flow)
    if df.empty:
        return

    # Pick a sensible x-axis label column
    flow_label = None
    for candidate in ("Runway", "A/D", "flow", "flow_name"):
        if candidate in df.columns:
            flow_label = candidate
            break
    x = df[flow_label].astype(str) if flow_label else df.index.astype(str)

    # Silhouette by flow
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, df["silhouette"].fillna(0.0))
    ax.set_ylabel("Silhouette (higher is better)")
    ax.set_title("Silhouette by flow")
    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_silhouette_by_flow.png", dpi=200)
    plt.close(fig)

    # Cluster count by flow
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, df["n_clusters"])
    ax.set_ylabel("# clusters")
    ax.set_title("Clusters by flow")
    fig.tight_layout()
    fig.savefig(plots_dir / "metrics_clusters_by_flow.png", dpi=200)
    plt.close(fig)

    if global_path.exists():
        g = pd.read_csv(global_path)
        if not g.empty:
            text = "\n".join(f"{k}: {v}" for k, v in g.iloc[0].to_dict().items())
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.axis("off")
            ax.text(0.01, 0.99, text, va="top", ha="left", family="monospace")
            ax.set_title("Global metrics")
            fig.tight_layout()
            fig.savefig(plots_dir / "metrics_global.png", dpi=200)
            plt.close(fig)


def _load_labels(exp_dir: Path) -> pd.DataFrame:
    parts = []
    for p in sorted(exp_dir.glob("labels_*.csv")):
        df = pd.read_csv(p)
        df["flow_file"] = p.stem.replace("labels_", "")
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _plot_cluster_trajectories(
    exp_dir: Path,
    plots_dir: Path,
    preprocessed_path: Path,
    max_flights_per_cluster: int,
) -> None:
    labels_df = _load_labels(exp_dir)
    if labels_df.empty:
        return

    needed_cols = ["flight_id", "step", "latitude", "longitude"]
    flow_keys = ["Runway"]
    cfg = _load_resolved_config(exp_dir)
    flow_keys = (cfg.get("flows", {}) or {}).get("flow_keys", flow_keys)
    needed_cols.extend([k for k in flow_keys if k not in needed_cols])

    pre = pd.read_csv(preprocessed_path, usecols=lambda c: c in set(needed_cols + ["A/D"]))
    pre = pre.merge(labels_df[["flight_id", "cluster_id"]], on="flight_id", how="inner")

    if flow_keys:
        flow_iter = pre.groupby(list(flow_keys))
    else:
        flow_iter = [(("ALL",), pre)]

    for flow_vals, df_flow in flow_iter:
        if not isinstance(flow_vals, tuple):
            flow_vals = (flow_vals,)
        flow_name = "_".join(str(v) for v in flow_vals) if flow_keys else "ALL"

        # Sample flights per cluster (omit noise from plotting, but log it)
        noise_count = int((df_flow["cluster_id"] == -1).sum())
        n_points = int(df_flow["step"].max() + 1) if "step" in df_flow else 0
        print(f"[plot] flow={flow_name} rows={len(df_flow)} noise_points={noise_count} n_points={n_points}")

        df_in = df_flow[df_flow["cluster_id"] != -1]
        if df_in.empty:
            continue

        flight_clusters = df_in[["flight_id", "cluster_id"]].drop_duplicates()
        chosen_ids = []
        for cid, grp in flight_clusters.groupby("cluster_id"):
            chosen_ids.extend(grp["flight_id"].head(max_flights_per_cluster).tolist())
        df_plot = df_in[df_in["flight_id"].isin(chosen_ids)].copy()

        clusters = sorted(df_plot["cluster_id"].unique())
        cmap = plt.get_cmap("tab10", max(len(clusters), 1))

        fig, ax = plt.subplots(figsize=(8, 6))
        for cid in clusters:
            subset = df_plot[df_plot["cluster_id"] == cid]
            for _, flight in subset.groupby("flight_id"):
                flight = flight.sort_values("step")
                ax.plot(flight["longitude"], flight["latitude"], color=cmap(int(cid) % 10), alpha=0.25, lw=0.8)
        ax.set_title(f"Clustered trajectories (flow={flow_name})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.text(0.01, 0.99, f"Noise points omitted: {noise_count}", transform=ax.transAxes, va="top", ha="left", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"clusters_{flow_name}.png", dpi=200)
        plt.close(fig)


def _plot_backbone_tracks(exp_dir: Path, plots_dir: Path, preprocessed_path: Path) -> None:
    labels_df = _load_labels(exp_dir)
    if labels_df.empty:
        return

    cfg = _load_resolved_config(exp_dir)
    backbone_cfg = cfg.get("backbone", {}) or {}
    percentiles = backbone_cfg.get("percentiles", [10, 50, 90])
    min_flights = int(backbone_cfg.get("min_flights_per_cluster", 10))

    # Load only needed columns for backbone computation
    cols = {"flight_id", "step", "latitude", "longitude", "altitude", "dist_to_airport_m"}
    for c in ("A/D", "Runway"):
        cols.add(c)
    df = pd.read_csv(preprocessed_path, usecols=lambda c: c in cols)
    df = df.merge(labels_df[["flight_id", "cluster_id"]], on="flight_id", how="inner")

    backbones = compute_backbones(df, percentiles=list(percentiles), min_flights_per_cluster=min_flights, use_utm=False)
    if backbones.empty:
        return
    backbones.to_csv(exp_dir / "backbone_tracks_generated.csv", index=False)

    # Plot per runway, per cluster
    for runway, group_rwy in backbones.groupby("Runway"):
        fig, ax = plt.subplots(figsize=(8, 6))
        clusters = sorted(group_rwy["cluster_id"].unique())
        cmap = plt.get_cmap("tab10", max(len(clusters), 1))
        for cid in clusters:
            g = group_rwy[group_rwy["cluster_id"] == cid]
            p50 = g[g["percentile"] == 50].sort_values("step")
            p10 = g[g["percentile"] == min(percentiles)].sort_values("step")
            p90 = g[g["percentile"] == max(percentiles)].sort_values("step")
            color = cmap(int(cid) % 10)
            if not p10.empty and not p90.empty and not p50.empty:
                ax.fill_between(p50["longitude"], p10["latitude"], p90["latitude"], color=color, alpha=0.12)
            ax.plot(p50["longitude"], p50["latitude"], color=color, lw=2, label=f"cluster {cid}")
        ax.set_title(f"Backbone tracks (Runway={runway})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"backbones_{runway}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot experiment results from output/experiments/<experiment>/")
    parser.add_argument("experiment", help="Experiment folder name under output/experiments/ (e.g., EXP77)")
    parser.add_argument("--output-root", default="output/experiments", help="Root experiments directory (default: output/experiments)")
    parser.add_argument(
        "--preprocessed",
        default=None,
        help="Override preprocessed CSV path. Otherwise uses config_resolved.yaml input.preprocessed_csv.",
    )
    parser.add_argument("--max-flights-per-cluster", type=int, default=50, help="Max trajectories per cluster to plot.")
    args = parser.parse_args()

    exp_dir = Path(args.output_root) / args.experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")

    plots_dir = _ensure_plots_dir(exp_dir)
    cfg = _load_resolved_config(exp_dir)
    preprocessed_path = args.preprocessed or (cfg.get("input", {}) or {}).get("preprocessed_csv")
    if preprocessed_path is None:
        raise FileNotFoundError(
            "Missing preprocessed CSV path. Pass --preprocessed or ensure config_resolved.yaml contains input.preprocessed_csv."
        )
    preprocessed_path = Path(preprocessed_path)
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {preprocessed_path}")

    _plot_metrics(exp_dir, plots_dir)
    _plot_cluster_trajectories(exp_dir, plots_dir, preprocessed_path, args.max_flights_per_cluster)
    _plot_backbone_tracks(exp_dir, plots_dir, preprocessed_path)
    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
