"""Export clustered flights to KML for a given experiment.

Usage:
  python scripts/export_experiment_kml.py --experiment EXP001 --max-per-cluster 200 --include-noise

Inputs:
  - output/experiments/<EXP>/labels_ALL.csv (must include flight_id, cluster_id, flow_label)
  - preprocessed CSV used in the experiment (from config_resolved.yaml)

Outputs:
  - output/experiments/<EXP>/kml/<EXP>_clusters.kml

Notes:
  - Coordinates are written as lon,lat,alt in WGS84.
  - Per-cluster downsampling is supported to keep KML size reasonable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pyproj import Transformer

try:
    from noise_simulation.generate_doc29_inputs import STARTPOINTS
except Exception:
    STARTPOINTS = {}


def _load_config_preprocessed(exp_dir: Path) -> Path:
    cfg_path = exp_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config_resolved.yaml at {cfg_path}")
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    preprocessed = cfg.get("input", {}).get("preprocessed_csv")
    if not preprocessed:
        raise ValueError("config_resolved.yaml missing input.preprocessed_csv")
    return Path(preprocessed)


def _kml_header(name: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
"""


def _kml_footer() -> str:
    return """  </Document>
</kml>
"""


def _placemark(name: str, description: str, coords: str, style: str | None = None) -> str:
    style_line = f"<styleUrl>#{style}</styleUrl>" if style else ""
    return f"""    <Placemark>
      <name>{name}</name>
      {style_line}
      <description><![CDATA[{description}]]></description>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>{coords}</coordinates>
      </LineString>
    </Placemark>
"""


def _point_placemark(name: str, description: str, lon: float, lat: float) -> str:
    return f"""    <Placemark>
      <name>{name}</name>
      <description><![CDATA[{description}]]></description>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>
"""


def _style_block(style_id: str, color: str) -> str:
    return f"""    <Style id="{style_id}">
      <LineStyle>
        <color>{color}</color>
        <width>2</width>
      </LineStyle>
    </Style>
"""


def _kml_color(hex_rgb: str, alpha: str = "ff") -> str:
    """Convert #RRGGBB to KML AABBGGRR."""
    hex_rgb = hex_rgb.lstrip("#")
    if len(hex_rgb) != 6:
        return "ff0000ff"
    r, g, b = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
    return f"{alpha}{b}{g}{r}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export experiment clusters to KML.")
    parser.add_argument("--experiment", required=True, help="Experiment name (e.g., EXP001).")
    parser.add_argument("--max-per-cluster", type=int, default=200, help="Max flights per cluster.")
    parser.add_argument("--include-noise", action="store_true", help="Include noise cluster (-1).")
    parser.add_argument("--include-runways", action="store_true", help="Add runway placemarks.")
    args = parser.parse_args()

    exp_dir = Path("output") / "experiments" / args.experiment
    labels_path = exp_dir / "labels_ALL.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels_ALL.csv at {labels_path}")

    preprocessed_path = _load_config_preprocessed(exp_dir)
    df_labels = pd.read_csv(labels_path)

    # Filter labels first and select per-cluster samples to reduce I/O.
    if not args.include_noise:
        df_labels = df_labels[df_labels["cluster_id"] != -1]
    if df_labels.empty:
        raise ValueError("No labels available after noise filtering.")

    # Sample flight_ids per (flow_label, cluster_id)
    sampled_ids = []
    for (flow_label, cluster_id), group in df_labels.groupby(["flow_label", "cluster_id"]):
        flight_ids = group["flight_id"].drop_duplicates().tolist()
        flight_ids = flight_ids[: args.max_per_cluster]
        sampled_ids.extend(flight_ids)
    sampled_ids = set(sampled_ids)

    # Read preprocessed CSV in chunks, keeping only sampled flight_ids.
    usecols = ["flight_id", "step", "latitude", "longitude", "altitude"]
    chunks = []
    for chunk in pd.read_csv(preprocessed_path, usecols=lambda c: c in usecols, chunksize=200_000):
        chunk = chunk[chunk["flight_id"].isin(sampled_ids)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No trajectory rows found for sampled flights.")
    df_traj = pd.concat(chunks, ignore_index=True)

    if "altitude" not in df_traj.columns:
        df_traj["altitude"] = 0.0

    df = df_traj.merge(df_labels, on="flight_id", how="inner")
    if not args.include_noise:
        df = df[df["cluster_id"] != -1]

    # Choose a small color palette, repeated as needed.
    palette = [
        "#E15759", "#59A14F", "#4E79A7", "#F28E2B",
        "#B07AA1", "#76B7B2", "#EDC948", "#FF9DA7",
    ]

    # Build styles per (flow_label, cluster_id) to avoid collisions across flows.
    style_lines = []
    styles = {}
    unique_groups = sorted(
        {(str(row.get("flow_label", "ALL")), int(row["cluster_id"])) for _, row in df.iterrows()}
    )
    for idx, (flow_label, cluster_id) in enumerate(unique_groups):
        style_id = f"cluster_{flow_label}_{cluster_id}".replace(" ", "_")
        if cluster_id == -1:
            color = _kml_color("#9E9E9E")
        else:
            color = _kml_color(palette[idx % len(palette)])
        styles[(flow_label, cluster_id)] = style_id
        style_lines.append(_style_block(style_id, color))

    out_dir = exp_dir / "kml"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.experiment}_clusters.kml"

    parts = [_kml_header(args.experiment), *style_lines]

    if args.include_runways and STARTPOINTS:
        transformer = Transformer.from_crs("epsg:32632", "epsg:4326", always_xy=True)
        runway_points = {}
        for (runway, mode), (x, y) in STARTPOINTS.items():
            lon, lat = transformer.transform(x, y)
            runway_points[(runway, mode)] = (lon, lat)
        for (runway, mode), (lon, lat) in sorted(runway_points.items()):
            parts.append(
                _point_placemark(
                    name=f"Runway {runway} ({mode})",
                    description=json.dumps({"runway": runway, "mode": mode}, indent=2),
                    lon=lon,
                    lat=lat,
                )
            )

    # Downsample per (flow_label, cluster_id)
    for (flow_label, cluster_id), group in df.groupby(["flow_label", "cluster_id"]):
        flight_ids = group["flight_id"].drop_duplicates().tolist()
        flight_ids = flight_ids[: args.max_per_cluster]
        for fid in flight_ids:
            flight = group[group["flight_id"] == fid].sort_values("step")
            coords = " ".join(
                f"{lon},{lat},{float(alt) if pd.notna(alt) else 0.0}"
                for lon, lat, alt in flight[["longitude", "latitude", "altitude"]].itertuples(index=False, name=None)
            )
            meta = flight.iloc[0]
            desc = {
                "flow_label": flow_label,
                "cluster_id": int(cluster_id),
                "A/D": meta.get("A/D"),
                "Runway": meta.get("Runway"),
                "aircraft_type_match": meta.get("aircraft_type_match"),
            }
            parts.append(
                _placemark(
                    name=f"flight_{fid}",
                    description=json.dumps(desc, indent=2),
                    coords=coords,
                    style=styles.get((flow_label, cluster_id)),
                )
            )

    parts.append(_kml_footer())
    out_path.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote KML to {out_path}")


if __name__ == "__main__":
    main()
