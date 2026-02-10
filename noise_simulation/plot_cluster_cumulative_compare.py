#!/usr/bin/env python
"""
Plot cumulative_res comparison between cluster subtracks and individual flights.

Inputs
  - EXP: experiment name (e.g., EXP001)
  - flow: flow label (e.g., Start_09L)
  - cluster: cluster id (e.g., 0)

Expected folder layout:
  noise_simulation/results/<EXP>/<flow>/cluster_<id>/<AIRCRAFT>/
    - groundtruth.csv   (cumulative_res for flights in the cluster)
    - subtracks.csv     (cumulative_res for cluster subtracks)

Outputs
  - PNG figure under:
    noise_simulation/results/<EXP>/<flow>/cluster_<id>/plots/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from math import ceil, sqrt

import pandas as pd
import matplotlib.pyplot as plt

try:
    from noise_simulation.automation.aircraft_types import NPD_TYPE_MAP, NPD_ID_TO_ICAO
except Exception:
    NPD_TYPE_MAP = {}
    NPD_ID_TO_ICAO = {}

# Traffic library aircraft type lookup (icao24 -> typecode).
try:
    from traffic.data import aircraft as traffic_aircraft
except Exception:
    traffic_aircraft = None

# Prefer canonical ICAO names when multiple map to the same NPD ID.
PREFERRED_ICAO_ORDER = [
    "B738",
    "A320",
    "A319",
    "A321",
    "A20N",
    "B38M",
    "B734",
    "E190",
    "E195",
    "E170",
    "E75L",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cumulative(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep=";")
    if "cumulative_res" not in df.columns:
        raise ValueError(f"Missing cumulative_res in {path}")
    return df["cumulative_res"].reset_index(drop=True)


def _load_cluster_typecodes(exp: str, flow: str, cluster_id: int) -> dict:
    """Return npd_id -> most common aircraft typecode for this flow/cluster."""
    labels_dir = _repo_root() / "output" / "experiments" / exp
    labels_path = labels_dir / f"labels_{flow}.csv"
    if not labels_path.exists():
        labels_path = labels_dir / "labels_ALL.csv"
    if not labels_path.exists():
        return {}

    usecols = ["cluster_id", "icao24", "flow_label"]
    labels = pd.read_csv(labels_path, usecols=usecols)
    labels = labels[labels["cluster_id"] == cluster_id]
    if "flow_label" in labels.columns:
        labels = labels[labels["flow_label"] == flow]
    if labels.empty:
        return {}

    if traffic_aircraft is None:
        return {}

    # Build icao24 -> typecode map from traffic DB.
    db = traffic_aircraft.data[["icao24", "typecode"]].dropna()
    db["icao24"] = db["icao24"].astype(str).str.upper()
    db["typecode"] = db["typecode"].astype(str).str.upper()
    icao_to_type = dict(zip(db["icao24"], db["typecode"]))

    labels["icao24"] = labels["icao24"].astype(str).str.upper()
    labels["typecode"] = labels["icao24"].map(icao_to_type)
    labels = labels[labels["typecode"].notna()]
    if labels.empty:
        return {}

    if not NPD_TYPE_MAP:
        return {}

    labels["npd_id"] = labels["typecode"].map(NPD_TYPE_MAP)
    labels = labels[labels["npd_id"].notna()]
    if labels.empty:
        return {}

    result = {}
    for npd_id, grp in labels.groupby("npd_id"):
        vc = grp["typecode"].value_counts()
        result[npd_id] = vc.index[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare subtracks vs flights cumulative_res.")
    parser.add_argument("--exp", required=True, help="Experiment name, e.g. EXP001")
    parser.add_argument("--flow", required=True, help="Flow label, e.g. Start_09L")
    parser.add_argument("--cluster", type=int, required=True, help="Cluster id, e.g. 0")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output PNG path. Defaults under cluster plots folder.",
    )
    args = parser.parse_args()

    base_dir = (
        _repo_root()
        / "noise_simulation"
        / "results"
        / args.exp
        / args.flow
        / f"cluster_{args.cluster}"
    )
    if not base_dir.exists():
        raise FileNotFoundError(f"Cluster folder not found: {base_dir}")

    aircraft_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    if not aircraft_dirs:
        raise FileNotFoundError(f"No aircraft folders under {base_dir}")

    series_by_aircraft = {}
    for aircraft_dir in aircraft_dirs:
        gt_path = aircraft_dir / "groundtruth.csv"
        st_path = aircraft_dir / "subtracks.csv"
        if not gt_path.exists() or not st_path.exists():
            continue
        gt = _load_cumulative(gt_path)
        st = _load_cumulative(st_path)
        series_by_aircraft[aircraft_dir.name] = (gt, st)

    if not series_by_aircraft:
        raise FileNotFoundError(f"No groundtruth/subtracks pairs found under {base_dir}")

    typecode_by_npd = _load_cluster_typecodes(args.exp, args.flow, args.cluster)

    n = len(series_by_aircraft)
    cols = ceil(sqrt(n))
    rows = ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    all_vals = []
    for gt, st in series_by_aircraft.values():
        all_vals.extend(gt.tolist())
        all_vals.extend(st.tolist())
    y_min = min(all_vals)
    y_max = max(all_vals)

    for idx, (aircraft, (gt, st)) in enumerate(series_by_aircraft.items()):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        x = list(range(1, len(gt) + 1))
        ax.plot(x, gt, marker="o", label="Flights (ground truth)")
        ax.plot(x, st, marker="s", label="Subtracks")
        # Prefer explicit ICAO prefix if present (e.g., A20N__V2527A).
        if "__" in aircraft:
            icao, _npd = aircraft.split("__", 1)
            title = icao
        else:
            npd = aircraft
            if npd in typecode_by_npd:
                title = typecode_by_npd[npd]
            elif NPD_TYPE_MAP:
                icao_list = [k for k, v in NPD_TYPE_MAP.items() if v == npd]
                if icao_list:
                    ordered = [k for k in PREFERRED_ICAO_ORDER if k in icao_list]
                    title = ordered[0] if ordered else sorted(icao_list)[0]
                else:
                    title = NPD_ID_TO_ICAO.get(npd, npd)
            else:
                title = NPD_ID_TO_ICAO.get(npd, npd)
        ax.set_title(title)
        ax.set_xlabel("Measurement point")
        ax.set_ylabel("cumulative_res")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    # Hide any empty subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.suptitle(f"{args.exp} {args.flow} cluster_{args.cluster}: cumulative_res comparison")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = base_dir / "plots" / f"cluster_{args.cluster}_cumulative_compare.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
