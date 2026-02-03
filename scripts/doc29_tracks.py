"""Generate Doc29-style 7-track overlays (backbone + 6 side tracks) for a flow.

Steps:
- Load a preprocessed CSV with x_utm/y_utm and step columns.
- Filter by A/D and Runway.
- Compute a backbone (median per step).
- Compute lateral offsets relative to the backbone tangent.
- Derive 7 tracks at offsets: 0, ±0.71S, ±1.43S, ±2.14S where S is the
  lateral std of offsets.
- Plot sampled trajectories, backbone, and side tracks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OFFSET_MULTIPLIERS = [0.0, 0.71, -0.71, 1.43, -1.43, 2.14, -2.14]
COLORS = {
    0.0: "black",
    0.71: "#1f77b4",
    -0.71: "#1f77b4",
    1.43: "#ff7f0e",
    -1.43: "#ff7f0e",
    2.14: "#2ca02c",
    -2.14: "#2ca02c",
}

TRACK_DEFS = [
    ("center", 0.0, 0.28),
    ("inner_left", 0.71, 0.22),
    ("inner_right", -0.71, 0.22),
    ("middle_left", 1.43, 0.11),
    ("middle_right", -1.43, 0.11),
    ("outer_left", 2.14, 0.03),
    ("outer_right", -2.14, 0.03),
]


def _backbone(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("step")[["x_utm", "y_utm"]].median().reset_index().sort_values("step")


def _tangent_vectors(bb: pd.DataFrame) -> np.ndarray:
    coords = bb[["x_utm", "y_utm"]].to_numpy()
    # central differences
    forward = np.vstack([coords[1:], coords[-1]])
    backward = np.vstack([coords[0], coords[:-1]])
    tangents = forward - backward
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tangents / norms


def _perp_vectors(tangents: np.ndarray) -> np.ndarray:
    return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)


def _lateral_offsets(df: pd.DataFrame, bb: pd.DataFrame, perps: np.ndarray) -> pd.Series:
    merged = df.merge(bb, on="step", suffixes=("", "_bb"))
    deltas = merged[["x_utm", "y_utm"]].to_numpy() - merged[["x_utm_bb", "y_utm_bb"]].to_numpy()
    offsets = np.einsum("ij,ij->i", deltas, perps[merged["step"].to_numpy()])
    return pd.Series(offsets, index=merged.index)


def _build_tracks(bb: pd.DataFrame, perps: np.ndarray, sigma: float) -> Dict[str, pd.DataFrame]:
    tracks: Dict[str, pd.DataFrame] = {}
    coords = bb[["x_utm", "y_utm"]].to_numpy()
    for name, mult, _weight in TRACK_DEFS:
        shift = mult * sigma * perps
        shifted = coords + shift
        tracks[name] = pd.DataFrame({"step": bb["step"], "x_utm": shifted[:, 0], "y_utm": shifted[:, 1]})
    return tracks


def _export_groundtrack(df: pd.DataFrame, out_path: Path) -> None:
    out = df.copy()
    out = out.rename(columns={"x_utm": "easting", "y_utm": "northing"})
    out.insert(0, "Unnamed: 0", range(len(out)))
    out[["Unnamed: 0", "easting", "northing"]].to_csv(out_path, sep=";", index=False)


def _export_tracks(tracks: Dict[str, pd.DataFrame], out_dir: Path, sigma: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for name, mult, weight in TRACK_DEFS:
        track_df = tracks[name]
        out_path = out_dir / f"groundtrack_{name}.csv"
        _export_groundtrack(track_df, out_path)
        manifest_rows.append(
            {
                "track": name,
                "multiplier": mult,
                "weight_fraction": weight,
                "sigma_m": sigma,
                "file": str(out_path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(out_dir / "tracks_manifest.csv", index=False)


def plot_doc29_tracks(
    df: pd.DataFrame,
    ad: str,
    runway: str,
    out_path: Path,
    sample_trajs: int = 300,
    export_dir: Path | None = None,
    plot: bool = True,
    cluster_id: int | None = None,
    cluster_label: str | None = None,
) -> None:
    df_flow = df[(df["A/D"] == ad) & (df["Runway"] == runway)].copy()
    if cluster_id is not None:
        if "cluster_id" not in df_flow.columns:
            raise ValueError("cluster_id filter requested but no cluster_id column found.")
        df_flow = df_flow[df_flow["cluster_id"] == cluster_id].copy()
    if df_flow.empty:
        suffix = f" cluster={cluster_id}" if cluster_id is not None else ""
        raise ValueError(f"No data for flow {ad}/{runway}{suffix}")
    bb = _backbone(df_flow)
    tangents = _tangent_vectors(bb)
    perps = _perp_vectors(tangents)
    offsets = _lateral_offsets(df_flow, bb, perps)
    sigma = float(offsets.std())
    tracks = _build_tracks(bb, perps, sigma)
    if export_dir is not None:
        _export_tracks(tracks, export_dir, sigma)

    if plot:
        # Plot sampled trajectories
        plt.figure(figsize=(8, 8))
        flight_ids = df_flow["flight_id"].dropna().unique()[:sample_trajs]
        for fid in flight_ids:
            sub = df_flow[df_flow["flight_id"] == fid]
            plt.plot(sub["x_utm"], sub["y_utm"], color="gray", alpha=0.15, linewidth=0.6)

        for name, mult, _weight in TRACK_DEFS:
            track_df = tracks[name]
            plt.plot(
                track_df["x_utm"],
                track_df["y_utm"],
                label=f"{name} ({mult:+.2f}σ)",
                color=COLORS.get(mult, "black"),
                linewidth=2.0 if mult == 0 else 1.2,
            )

        title = f"Doc29 7-Track Layout: {ad} {runway}"
        if cluster_label:
            title = f"{title} ({cluster_label})"
        plt.title(title)
        plt.axis("equal")
        plt.legend()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Wrote {out_path} (σ={sigma:.2f} m lateral)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Doc29 backbone + 6 side tracks for a flow.")
    parser.add_argument("--preprocessed", required=True, help="Path to preprocessed CSV with x_utm/y_utm/step/flight_id/A/D/Runway.")
    parser.add_argument("--ad", help="A/D value to filter (e.g., Start or Landung).")
    parser.add_argument("--runway", help="Runway to filter (e.g., 09R).")
    parser.add_argument("--all-flows", action="store_true", help="Generate tracks for every A/D + Runway combination.")
    parser.add_argument("--out", default="output/eda/doc29_tracks.png", help="Output PNG path.")
    parser.add_argument("--sample-trajs", type=int, default=300, help="Number of trajectories to sample for background plot.")
    parser.add_argument("--export-dir", default=None, help="Optional directory to write groundtrack_*.csv outputs.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting; export only.")
    parser.add_argument("--labels", default=None, help="Optional labels csv or experiment folder with labels_*.csv.")
    parser.add_argument("--cluster-id", type=int, default=None, help="Use a specific cluster id (requires --labels).")
    parser.add_argument(
        "--cluster-strategy",
        choices=["dominant", "all"],
        default="dominant",
        help="Cluster selection when --cluster-id is not set (default: dominant).",
    )
    parser.add_argument("--include-noise", action="store_true", help="Include noise cluster (-1) when using --cluster-strategy all.")
    args = parser.parse_args()

    df = pd.read_csv(args.preprocessed)
    for col in ("x_utm", "y_utm", "step", "flight_id", "A/D", "Runway"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.preprocessed}")

    labels_df: pd.DataFrame | None = None
    if args.labels:
        labels_path = Path(args.labels)
        if labels_path.is_dir():
            parts = []
            for p in sorted(labels_path.glob("labels_*.csv")):
                parts.append(pd.read_csv(p))
            labels_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        elif labels_path.suffix == ".csv":
            labels_df = pd.read_csv(labels_path)
        else:
            raise ValueError("labels path must be a directory or .csv file.")

        if labels_df.empty:
            raise ValueError("No labels loaded; check --labels path.")
        if "cluster_id" not in labels_df.columns or "flight_id" not in labels_df.columns:
            raise ValueError("Labels must include flight_id and cluster_id columns.")
        df = df.merge(labels_df[["flight_id", "cluster_id"]], on="flight_id", how="inner")

    export_dir = Path(args.export_dir) if args.export_dir else None
    plot = not args.no_plot

    def pick_clusters(flow_df: pd.DataFrame) -> list[int]:
        if args.cluster_id is not None:
            return [args.cluster_id]
        if args.cluster_strategy == "all":
            clusters = sorted(flow_df["cluster_id"].dropna().unique()) if "cluster_id" in flow_df.columns else []
            if not args.include_noise:
                clusters = [cid for cid in clusters if cid != -1]
            return clusters
        # dominant
        if "cluster_id" not in flow_df.columns:
            return []
        counts = flow_df[flow_df["cluster_id"] != -1].groupby("cluster_id").size()
        if counts.empty:
            return []
        dominant = int(counts.sort_values(ascending=False).index[0])
        return [dominant]

    if args.all_flows:
        flows = df[["A/D", "Runway"]].dropna().drop_duplicates()
        for _, row in flows.iterrows():
            ad = row["A/D"]
            runway = row["Runway"]
            flow_out = Path(args.out)
            if export_dir is not None:
                flow_export = export_dir / f"{ad}_{runway}"
            else:
                flow_export = None

            flow_df = df[(df["A/D"] == ad) & (df["Runway"] == runway)]
            cluster_ids = pick_clusters(flow_df) if labels_df is not None else [None]
            for cid in cluster_ids:
                cluster_suffix = f"_cluster{cid}" if cid is not None else ""
                out_path = flow_out.with_name(f"{flow_out.stem}_{ad}_{runway}{cluster_suffix}{flow_out.suffix}")
                export_path = flow_export / f"cluster_{cid}" if (flow_export and cid is not None) else flow_export
                plot_doc29_tracks(
                    df,
                    ad,
                    runway,
                    out_path,
                    sample_trajs=args.sample_trajs,
                    export_dir=export_path,
                    plot=plot,
                    cluster_id=cid,
                    cluster_label=f"cluster {cid}" if cid is not None else None,
                )
    else:
        if not args.ad or not args.runway:
            raise ValueError("Provide --ad and --runway, or use --all-flows.")
        flow_df = df[(df["A/D"] == args.ad) & (df["Runway"] == args.runway)]
        cluster_ids = pick_clusters(flow_df) if labels_df is not None else [None]
        for cid in cluster_ids:
            cluster_suffix = f"_cluster{cid}" if cid is not None else ""
            out_path = Path(args.out)
            if cid is not None:
                out_path = out_path.with_name(f"{out_path.stem}{cluster_suffix}{out_path.suffix}")
            export_path = export_dir / f"cluster_{cid}" if (export_dir and cid is not None) else export_dir
            plot_doc29_tracks(
                df,
                args.ad,
                args.runway,
                out_path,
                sample_trajs=args.sample_trajs,
                export_dir=export_path,
                plot=plot,
                cluster_id=cid,
                cluster_label=f"cluster {cid}" if cid is not None else None,
            )


if __name__ == "__main__":
    main()
