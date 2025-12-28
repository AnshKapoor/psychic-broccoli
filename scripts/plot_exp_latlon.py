"""Plot clustered trajectories in latitude/longitude for a given experiment.

Usage:
  python scripts/plot_exp_latlon.py EXP30_kmeans6_optics_refine
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _load_config(exp_dir: Path) -> dict:
    cfg_path = exp_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _read_parquet_cols(path: Path, cols: list[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=cols)
    except Exception:
        return pd.read_parquet(path)


def _load_labels(exp_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    cols = ["flight_id", "cluster_id", "A/D", "Runway", "icao24"]
    for p in sorted(exp_dir.glob("labels_*.parquet")):
        parts.append(_read_parquet_cols(p, cols))
    if not parts:
        labels_all = exp_dir / "labels_ALL.parquet"
        if labels_all.exists():
            parts.append(_read_parquet_cols(labels_all, cols))
    if not parts:
        return pd.DataFrame()
    labels = pd.concat(parts, ignore_index=True)
    labels["flight_id"] = labels["flight_id"].astype(int)
    labels["cluster_id"] = labels["cluster_id"].astype(int)
    return labels.drop_duplicates(subset=["flight_id"])


def _sample_flight_ids(labels: pd.DataFrame, max_per_cluster: int, seed: int) -> set[int]:
    rng = np.random.default_rng(seed)
    sampled: list[int] = []
    for cid in sorted(labels["cluster_id"].unique()):
        if cid == -1:
            continue
        ids = labels.loc[labels["cluster_id"] == cid, "flight_id"].to_numpy()
        if len(ids) > max_per_cluster:
            ids = rng.choice(ids, size=max_per_cluster, replace=False)
        sampled.extend(ids.tolist())
    return set(sampled)


def _read_preprocessed(preprocessed_path: Path, flight_ids: set[int]) -> pd.DataFrame:
    usecols = ["flight_id", "step", "latitude", "longitude"]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(preprocessed_path, usecols=usecols, chunksize=200_000):
        chunk["flight_id"] = chunk["flight_id"].astype(int)
        chunk = chunk[chunk["flight_id"].isin(flight_ids)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _cluster_palette(cluster_ids: list[int]) -> dict[int, str]:
    tableau10 = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]
    palette = {}
    for idx, cid in enumerate(cluster_ids):
        palette[cid] = tableau10[idx % len(tableau10)]
    return palette


def _load_aircraft_type_map(
    enhanced_glob: str,
    target_icao24: set[str],
    cache_path: Path | None,
    chunk_size: int = 200_000,
) -> dict[str, str]:
    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if {"icao24", "aircraft_type"}.issubset(cached.columns):
            cached["icao24"] = cached["icao24"].astype(str).str.upper()
            cached["aircraft_type"] = cached["aircraft_type"].astype(str)
            return dict(zip(cached["icao24"], cached["aircraft_type"]))

    files = sorted(Path().glob(enhanced_glob))
    if not files:
        return {}

    type_map: dict[str, str] = {}
    priority: dict[str, int] = {}
    usecols = ["icao24", "aircraft_type_adsb", "aircraft_type_noise"]

    for path in files:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
            chunk = chunk[chunk["icao24"].notna()]
            if chunk.empty:
                continue
            chunk["icao24"] = chunk["icao24"].astype(str).str.upper()
            chunk = chunk[chunk["icao24"].isin(target_icao24)]
            if chunk.empty:
                continue

            adsb = chunk[chunk["aircraft_type_adsb"].notna()]
            if not adsb.empty:
                adsb = adsb.drop_duplicates("icao24")
                for icao, t in adsb[["icao24", "aircraft_type_adsb"]].itertuples(index=False):
                    t = str(t).strip().upper()
                    if not t:
                        continue
                    if priority.get(icao, 0) < 2:
                        type_map[icao] = t
                        priority[icao] = 2

            noise = chunk[chunk["aircraft_type_noise"].notna()]
            if not noise.empty:
                noise = noise.drop_duplicates("icao24")
                for icao, t in noise[["icao24", "aircraft_type_noise"]].itertuples(index=False):
                    t = str(t).strip().upper()
                    if not t:
                        continue
                    if priority.get(icao, 0) < 1:
                        type_map[icao] = t
                        priority[icao] = 1

        print(f"[aircraft types] {path.name}: mapped {len(type_map)} / {len(target_icao24)}")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"icao24": list(type_map.keys()), "aircraft_type": list(type_map.values())}
        ).to_csv(cache_path, index=False)

    return type_map


def _plot_cluster_composition(
    labels_df: pd.DataFrame,
    category_col: str,
    out_path: Path,
    title: str,
    top_n: int | None = None,
) -> None:
    df = labels_df[labels_df["cluster_id"] != -1].copy()
    df[category_col] = df[category_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    if top_n:
        counts = df[category_col].value_counts()
        keep = set(counts.head(top_n).index)
        df[category_col] = df[category_col].where(df[category_col].isin(keep), "Other")

    pivot = df.groupby(["cluster_id", category_col]).size().unstack(fill_value=0)
    pivot = pivot.sort_index()

    cmap = plt.get_cmap("tab20", max(pivot.shape[1], 1))
    colors = [cmap(i) for i in range(pivot.shape[1])]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.85, edgecolor="none")

    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Flights")
    ax.set_title(title)
    ax.legend(
        title=category_col.replace("_", " ").title(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
    )
    ax.grid(axis="y", color="#e9e9e9")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_cluster_counts(
    labels_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    df = labels_df[labels_df["cluster_id"] != -1].copy()
    counts = df["cluster_id"].value_counts().sort_index()
    total = counts.sum()
    if total == 0:
        return
    perc = (counts / total * 100.0).round(1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#4E79A7", alpha=0.9)

    for bar, p in zip(bars, perc.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{p}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Flights")
    ax.set_title(title)
    ax.grid(axis="y", color="#e9e9e9")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_trajectories(
    df: pd.DataFrame,
    palette: dict[int, str],
    out_path: Path,
    title: str,
    max_flights_per_cluster: int,
) -> None:
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(10, 8))
    for cid in sorted(palette.keys()):
        subset = df[df["cluster_id"] == cid]
        for _, flight in subset.groupby("flight_id"):
            flight = flight.sort_values("step")
            ax.plot(
                flight["longitude"],
                flight["latitude"],
                color=palette[cid],
                alpha=0.35,
                lw=0.8,
            )

    ax.set_title(f"{title}\nSampled trajectories per cluster (max {max_flights_per_cluster})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e9e9e9", linewidth=0.7)

    handles = [
        Line2D([0], [0], color=palette[cid], lw=2, label=f"cluster {cid}")
        for cid in sorted(palette.keys())
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_points(
    df: pd.DataFrame,
    palette: dict[int, str],
    out_path: Path,
    title: str,
    max_points_per_cluster: int,
    seed: int,
) -> None:
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(10, 8))
    rng = np.random.default_rng(seed)
    for cid in sorted(palette.keys()):
        subset = df[df["cluster_id"] == cid]
        if len(subset) > max_points_per_cluster:
            subset = subset.sample(max_points_per_cluster, random_state=int(rng.integers(0, 1_000_000)))
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            s=6,
            alpha=0.45,
            color=palette[cid],
            label=f"cluster {cid}",
        )

    ax.set_title(f"{title}\nSampled points per cluster (max {max_points_per_cluster})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e9e9e9", linewidth=0.7)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, markerscale=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot clustered trajectories in latitude/longitude.")
    parser.add_argument("experiment", help="Experiment folder name under output/")
    parser.add_argument("--output-root", default="output", help="Root output directory")
    parser.add_argument("--preprocessed", default=None, help="Override preprocessed CSV path")
    parser.add_argument("--enhanced-glob", default="Enhanced/matched_trajs_*.csv")
    parser.add_argument("--aircraft-type-top-n", type=int, default=10)
    parser.add_argument("--max-flights-per-cluster", type=int, default=120)
    parser.add_argument("--max-points-per-cluster", type=int, default=25000)
    parser.add_argument("--min-points-per-flight", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exp_dir = Path(args.output_root) / args.experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")

    labels = _load_labels(exp_dir)
    if labels.empty:
        raise FileNotFoundError("No labels_*.parquet or labels_ALL.parquet found.")

    cfg = _load_config(exp_dir)
    preprocessed_path = args.preprocessed or (cfg.get("input", {}) or {}).get("preprocessed_csv")
    if preprocessed_path is None:
        raise FileNotFoundError(
            "Missing preprocessed CSV path. Pass --preprocessed or ensure config_resolved.yaml contains input.preprocessed_csv."
        )
    preprocessed_path = Path(preprocessed_path)
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {preprocessed_path}")

    sampled_ids = _sample_flight_ids(labels, args.max_flights_per_cluster, args.seed)
    df = _read_preprocessed(preprocessed_path, sampled_ids)
    if df.empty:
        raise ValueError("No rows loaded from preprocessed CSV after sampling.")

    df = df.merge(labels, on="flight_id", how="left")
    df = df[df["cluster_id"] != -1]

    if args.min_points_per_flight > 1:
        flight_counts = df.groupby("flight_id").size()
        keep_ids = flight_counts[flight_counts >= args.min_points_per_flight].index
        df = df[df["flight_id"].isin(keep_ids)]
        if df.empty:
            raise ValueError(
                "No flights left after min-points filtering. "
                "Lower --min-points-per-flight or increase sampling."
            )
        print(f"Kept {len(keep_ids)} flights with >= {args.min_points_per_flight} points.")

    cluster_ids = sorted(df["cluster_id"].unique())
    palette = _cluster_palette(cluster_ids)

    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    title = f"{exp_dir.name} clusters (lat/lon)"

    _plot_trajectories(
        df,
        palette,
        plots_dir / f"{exp_dir.name}_clustered_trajectories_latlon.png",
        title,
        args.max_flights_per_cluster,
    )
    _plot_points(
        df,
        palette,
        plots_dir / f"{exp_dir.name}_clustered_points_latlon.png",
        title,
        args.max_points_per_cluster,
        args.seed,
    )

    # Composition plots (cluster vs runway, A/D, aircraft type)
    labels_full = labels.copy()
    labels_full["icao24"] = labels_full["icao24"].astype(str).str.upper()

    _plot_cluster_composition(
        labels_full,
        "Runway",
        plots_dir / f"{exp_dir.name}_cluster_by_runway.png",
        f"{exp_dir.name}: Runway mix per cluster",
    )

    _plot_cluster_composition(
        labels_full,
        "A/D",
        plots_dir / f"{exp_dir.name}_cluster_by_ad.png",
        f"{exp_dir.name}: Arrival/Departure mix per cluster",
    )

    _plot_cluster_counts(
        labels_full,
        plots_dir / f"{exp_dir.name}_cluster_counts.png",
        f"{exp_dir.name}: Flights per cluster (%)",
    )

    cache_path = exp_dir / "aircraft_type_by_icao24.csv"
    icao24_set = set(labels_full["icao24"].dropna().astype(str).str.upper())
    type_map = _load_aircraft_type_map(args.enhanced_glob, icao24_set, cache_path)
    if type_map:
        labels_full["aircraft_type"] = labels_full["icao24"].map(type_map)
        _plot_cluster_composition(
            labels_full,
            "aircraft_type",
            plots_dir / f"{exp_dir.name}_cluster_by_aircraft_type.png",
            f"{exp_dir.name}: Aircraft type mix per cluster (top {args.aircraft_type_top_n})",
            top_n=args.aircraft_type_top_n,
        )
    else:
        print("No aircraft type mapping found; skipping aircraft type plot.")

    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
