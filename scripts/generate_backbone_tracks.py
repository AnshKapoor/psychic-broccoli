"""Build backbone tracks for noise modelling (ECAC Doc 29 style).

The script aggregates individual matched trajectories into representative
backbone tracks per flow (e.g., arrival/departure and runway). It smooths each
trajectory, resamples it to a fixed length, and then computes per-step
percentiles so noise models can consume a compact, robust centreline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

# Ensure project root is on sys.path for module resolution when launched directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.trajectory_preprocessing import preprocess_trajectory


@dataclass
class BackboneConfig:
    """Configuration for backbone track generation."""

    input_csv: Path
    output_csv: Path = Path("reports/backbone_tracks.csv")
    group_keys: Sequence[str] = ("MP", "t_ref", "icao24")
    backbone_keys: Sequence[str] = ("A/D", "Runway")
    numeric_columns: Sequence[str] = (
        "latitude",
        "longitude",
        "altitude",
        "groundspeed",
        "vertical_rate",
        "track",
        "dist_to_airport_m",
        "geoaltitude",
    )
    required_columns: Sequence[str] = (
        "timestamp",
        "A/D",
        "Runway",
        "latitude",
        "longitude",
        "altitude",
        "groundspeed",
        "track",
        "dist_to_airport_m",
    )
    target_length: int = 20
    min_group_size: int = 10
    # Test-mode controls for lightweight dry-runs
    test_mode: bool = False
    test_max_rows: int = 5000
    test_max_flows: int = 3
    test_min_group_size: int = 2


def load_config(path: Path) -> BackboneConfig:
    """Load YAML into a :class:`BackboneConfig`."""

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return BackboneConfig(
        input_csv=Path(raw["input_csv"]),
        output_csv=Path(raw.get("output_csv", "reports/backbone_tracks.csv")),
        group_keys=tuple(raw.get("group_keys", ("MP", "t_ref", "icao24"))),
        backbone_keys=tuple(raw.get("backbone_keys", ("A/D", "Runway"))),
        numeric_columns=tuple(
            raw.get(
                "numeric_columns",
                [
                    "latitude",
                    "longitude",
                    "altitude",
                    "groundspeed",
                    "vertical_rate",
                    "track",
                    "dist_to_airport_m",
                    "geoaltitude",
                ],
            )
        ),
        required_columns=tuple(
            raw.get(
                "required_columns",
                [
                    "timestamp",
                    "A/D",
                    "Runway",
                    "latitude",
                    "longitude",
                    "altitude",
                    "groundspeed",
                    "track",
                    "dist_to_airport_m",
                ],
            )
        ),
        target_length=int(raw.get("target_length", 20)),
        min_group_size=int(raw.get("min_group_size", 10)),
        test_mode=bool(raw.get("test_mode", False)),
        test_max_rows=int(raw.get("test_max_rows", 5000)),
        test_max_flows=int(raw.get("test_max_flows", 3)),
        test_min_group_size=int(raw.get("test_min_group_size", 2)),
    )


def preprocess_input(df: pd.DataFrame, cfg: BackboneConfig) -> pd.DataFrame:
    """Normalise types and enforce required columns before grouping."""

    df_proc = df.copy()
    missing = [col for col in cfg.required_columns if col not in df_proc.columns]
    if missing:
        raise KeyError(f"Input CSV missing required columns: {missing}")

    df_proc["timestamp"] = pd.to_datetime(df_proc["timestamp"], errors="coerce")
    df_proc["Runway"] = df_proc["Runway"].astype(str).str.upper().str.strip().replace({"nan": pd.NA})
    if "A/D" in df_proc.columns:
        df_proc["A/D"] = df_proc["A/D"].astype(str).str.strip()

    for col in cfg.numeric_columns:
        if col in df_proc.columns:
            df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")

    df_proc = df_proc.dropna(subset=cfg.required_columns)
    return df_proc


def _iter_processed_groups(
    df: pd.DataFrame, cfg: BackboneConfig
) -> Iterable[tuple[pd.Series, pd.DataFrame]]:
    """Yield (metadata, processed trajectory) per flight."""

    for group_values, group_df in df.groupby(list(cfg.group_keys)):
        if len(group_df) < 3:
            continue
        processed = preprocess_trajectory(group_df, target_length=cfg.target_length)
        if processed.empty:
            continue
        meta = pd.Series(group_values, index=cfg.group_keys)
        for key in cfg.backbone_keys:
            if key in group_df.columns:
                meta[key] = group_df[key].iloc[0]
        yield meta, processed


def _percentiles(values: np.ndarray) -> Tuple[float, float, float]:
    """Return (p10, p50, p90) for a 1D array."""

    return (
        float(np.nanpercentile(values, 10)),
        float(np.nanmedian(values)),
        float(np.nanpercentile(values, 90)),
    )


def build_backbone_tracks(
    processed_groups: Iterable[tuple[pd.Series, pd.DataFrame]],
    cfg: BackboneConfig,
) -> pd.DataFrame:
    """Aggregate resampled trajectories into backbone tracks."""

    grouped: Dict[Tuple[str, ...], List[pd.DataFrame]] = {}

    for meta, traj in processed_groups:
        key = tuple(meta[key] for key in cfg.backbone_keys)
        grouped.setdefault(key, []).append(traj)

    backbone_rows: List[Dict[str, object]] = []
    for key, trajectories in grouped.items():
        if len(trajectories) < cfg.min_group_size:
            continue

        stack = {
            col: np.vstack([t[col].to_numpy() for t in trajectories if col in t.columns])
            for col in cfg.numeric_columns
            if any(col in t.columns for t in trajectories)
        }

        for step in range(cfg.target_length):
            row: Dict[str, object] = {cfg.backbone_keys[i]: key[i] for i in range(len(cfg.backbone_keys))}
            row["step"] = step
            row["n_flights"] = len(trajectories)
            for col, values in stack.items():
                if values.shape[1] <= step:
                    continue
                p10, p50, p90 = _percentiles(values[:, step])
                row[f"{col}_p10"] = p10
                row[f"{col}_p50"] = p50
                row[f"{col}_p90"] = p90
            backbone_rows.append(row)

    if not backbone_rows:
        return pd.DataFrame()

    return pd.DataFrame(backbone_rows)


def parse_args() -> argparse.Namespace:
    """Return CLI arguments."""

    parser = argparse.ArgumentParser(description="Generate backbone tracks for noise modelling.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/backbone_legacy.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable lightweight test mode (row/flow caps) to validate the pipeline end-to-end.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""

    args = parse_args()
    cfg = load_config(args.config)

    df = pd.read_csv(cfg.input_csv)
    test_mode = args.test or cfg.test_mode
    if test_mode:
        df = df.head(cfg.test_max_rows)
    df = preprocess_input(df, cfg)

    if test_mode:
        flows = df[list(cfg.backbone_keys)].dropna().drop_duplicates().head(cfg.test_max_flows)
        if not flows.empty:
            df = df.merge(flows, on=list(cfg.backbone_keys), how="inner")
        cfg.min_group_size = min(cfg.min_group_size, cfg.test_min_group_size)

    processed = list(_iter_processed_groups(df, cfg))
    backbone_df = build_backbone_tracks(processed, cfg)

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    backbone_df.to_csv(cfg.output_csv, index=False)
    if backbone_df.empty:
        raise ValueError(
            "No backbone tracks generated. Likely causes: all flight groups shorter than 3 points, "
            "min_group_size too high, or filters dropped required columns."
        )

    flows = backbone_df[list(cfg.backbone_keys)].drop_duplicates().shape[0]
    suffix = " [test mode]" if test_mode else ""
    print(f"Backbone tracks written to {cfg.output_csv}{suffix} (flows: {flows}, rows: {len(backbone_df)})")


if __name__ == "__main__":
    main()
