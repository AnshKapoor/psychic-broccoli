"""Experiment runner for extended clustering and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustering.distances import build_feature_matrix, pairwise_distance_matrix
from clustering.evaluation import compute_internal_metrics
from clustering.registry import get_clusterer


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def filter_flows(df: pd.DataFrame, flow_keys: List[str], include: List[List[str]] | None) -> pd.DataFrame:
    if not include or not flow_keys:
        return df
    include_set = {tuple(item) for item in include}
    mask = df[flow_keys].apply(tuple, axis=1).isin(include_set)
    return df[mask].reset_index(drop=True)


def weighted_mean(values: List[float], weights: List[int]) -> float:
    vals = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    return float(np.average(vals, weights=w))


def _latest_preprocessed() -> Path | None:
    csv_dir = Path("output") / "run" / "csv"
    if not csv_dir.exists():
        return None
    candidates = sorted(csv_dir.glob("preprocessed_*.csv"))
    return candidates[-1] if candidates else None


def run_experiment(cfg_path: Path, preprocessed_override: Path | None = None) -> None:
    cfg = load_config(cfg_path)
    clustering_cfg: Dict[str, object] = cfg.get("clustering", {}) or {}
    eval_cfg: Dict[str, object] = clustering_cfg.get("evaluation", {}) or {}
    method = clustering_cfg.get("method", "optics")
    distance_metric = clustering_cfg.get("distance_metric", "euclidean")
    distance_params = clustering_cfg.get("distance_params", {}) or {}
    experiment_name = cfg.get("output", {}).get("experiment_name", "experiment")

    flows_cfg = cfg.get("flows", {}) or {}
    flow_keys = list(flows_cfg.get("flow_keys") or [])
    include_flows = flows_cfg.get("include", []) or []

    input_cfg = cfg.get("input", {}) or {}
    preprocessed_csv = preprocessed_override or input_cfg.get("preprocessed_csv")
    if preprocessed_csv is None:
        latest = _latest_preprocessed()
        if latest is None:
            raise FileNotFoundError(
                "No preprocessed CSV specified and none found under output/run/csv. "
                "Set input.preprocessed_csv in the config or pass --preprocessed."
            )
        preprocessed_csv = latest
    df = pd.read_csv(preprocessed_csv)
    df = filter_flows(df, flow_keys, include_flows)

    output_dir = Path(cfg.get("output", {}).get("dir", "output")) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare outputs
    metrics_rows: List[dict] = []
    label_paths: List[Path] = []
    log_lines: List[str] = []

    # Build experiment log header (human-readable).
    log_lines.append(f"Experiment: {experiment_name}")
    log_lines.append(f"Config: {cfg_path}")
    log_lines.append(f"Run started (UTC): {datetime.now(timezone.utc).isoformat()}")
    log_lines.append(f"Method: {method}")
    log_lines.append(f"Distance metric: {distance_metric}")
    log_lines.append(f"Preprocessed CSV: {preprocessed_csv}")
    vector_cols = cfg.get("features", {}).get("vector_cols", ["x_utm", "y_utm"])
    log_lines.append(f"Vector columns: {vector_cols}")
    resample_n = cfg.get("preprocessing", {}).get("resampling", {}).get("n_points")
    if resample_n is not None:
        log_lines.append(f"Resampling n_points: {resample_n}")
    log_lines.append(f"Flow keys: {flow_keys if flow_keys else ['GLOBAL']}")

    method_params = clustering_cfg.get(method, {}) or {}
    if method == "two_stage":
        stage1 = method_params.get("stage1_method")
        stage2 = method_params.get("stage2_method")
        stage1_params = method_params.get("stage1_params", {})
        stage2_params = method_params.get("stage2_params", {})
        log_lines.append(f"Stage1: {stage1} params={stage1_params}")
        log_lines.append(f"Stage2: {stage2} params={stage2_params}")
    else:
        log_lines.append(f"Params: {method_params}")
    log_lines.append("")

    # Per-flow clustering (or global if no flow_keys provided)
    if flow_keys:
        flow_iter = df.groupby(flow_keys)
    else:
        flow_iter = [(("ALL",), df)]

    for flow_vals, flow_df in flow_iter:
        if not isinstance(flow_vals, tuple):
            flow_vals = (flow_vals,)
        flow_name = "_".join(str(v) for v in flow_vals) if flow_keys else "ALL"

        # Build features/trajectories
        vector_cols = cfg.get("features", {}).get("vector_cols", ["x_utm", "y_utm"])
        X, trajs = build_feature_matrix(flow_df, vector_cols=vector_cols)

        clusterer = get_clusterer(method)
        precomputed_needed = distance_metric in {"dtw", "frechet"}
        if precomputed_needed and not clusterer.supports_precomputed:
            raise ValueError(f"{method} does not support precomputed distances ({distance_metric}).")

        if precomputed_needed:
            params = {
                "distance_metric": distance_metric,
                "n_points": cfg.get("preprocessing", {}).get("resampling", {}).get("n_points"),
            }
            params.update(distance_params)
            D = pairwise_distance_matrix(
                trajs if distance_metric in {"dtw", "frechet"} else X,
                metric=distance_metric,
                cache_dir=output_dir / "cache",
                flow_name=flow_name,
                params=params,
            )
            labels = clusterer.fit_predict(D, metric="precomputed")
            metrics = compute_internal_metrics(D, labels, metric_mode="precomputed", include_noise=eval_cfg.get("include_noise", False))
        else:
            labels = clusterer.fit_predict(
                X,
                **(clustering_cfg.get(method, {}) or {}),
            )
            metrics = compute_internal_metrics(X, labels, metric_mode="features", include_noise=eval_cfg.get("include_noise", False))
            if method == "gmm" and getattr(clusterer, "last_model", None) is not None:
                model = clusterer.last_model
                metrics["bic"] = float(model.bic(X))
                metrics["aic"] = float(model.aic(X))

        # Log per-flow details.
        flight_counts = flow_df.groupby("flight_id").size()
        n_points_min = int(flight_counts.min()) if not flight_counts.empty else 0
        n_points_med = int(flight_counts.median()) if not flight_counts.empty else 0
        n_points_max = int(flight_counts.max()) if not flight_counts.empty else 0

        if flow_keys:
            flow_label = "_".join(str(v) for v in flow_vals)
        else:
            flow_label = "GLOBAL"

        log_lines.append(f"Flow: {flow_label}")
        log_lines.append(f"  Flights: {len(labels)}")
        log_lines.append(f"  Points: total={len(flow_df)} per_flight[min/med/max]={n_points_min}/{n_points_med}/{n_points_max}")
        log_lines.append(f"  Metrics: n_clusters={metrics.get('n_clusters')} noise_frac={metrics.get('noise_frac')}")
        log_lines.append(
            f"  Metrics: silhouette={metrics.get('silhouette')} davies_bouldin={metrics.get('davies_bouldin')} calinski_harabasz={metrics.get('calinski_harabasz')}"
        )

        counts = pd.Series(labels).value_counts().sort_index()
        counts_str = ", ".join(f"{int(k)}={int(v)}" for k, v in counts.items())
        log_lines.append(f"  Cluster counts: {counts_str}")

        if flow_keys:
            metrics.update({key: val for key, val in zip(flow_keys, flow_vals)})
        else:
            metrics["flow"] = "ALL"
        metrics["n_flights"] = len(labels)
        metrics_rows.append(metrics)

        meta_cols = ["flight_id"]
        for col in (*flow_keys, "A/D", "Runway", "icao24", "callsign"):
            if col in flow_df.columns and col not in meta_cols:
                meta_cols.append(col)
        labeled = flow_df[meta_cols].drop_duplicates(subset=["flight_id"]).copy()
        labeled["cluster_id"] = labels
        label_path = output_dir / f"labels_{flow_name}.parquet"
        labeled.to_parquet(label_path, index=False)
        label_paths.append(label_path)
        log_lines.append(f"  Labels: {label_path}")
        log_lines.append("")

    # Cluster/runway breakdown (helpful when clustering across all flows)
    labels_all = []
    for p in label_paths:
        df_lab = pd.read_parquet(p)
        labels_all.append(df_lab)
    if labels_all:
        df_lab_all = pd.concat(labels_all, ignore_index=True)
        if "Runway" in df_lab_all.columns:
            counts = (
                df_lab_all.groupby(["cluster_id", "Runway"])
                .size()
                .reset_index(name="n_flights")
                .sort_values(["cluster_id", "n_flights"], ascending=[True, False])
            )
            counts.to_csv(output_dir / "cluster_runway_counts.csv", index=False)

    # Aggregate metrics globally (weighted by flights)
    if metrics_rows:
        df_metrics = pd.DataFrame(metrics_rows)
        df_metrics.to_csv(output_dir / "metrics_by_flow.csv", index=False)
        weight = df_metrics["n_flights"]
        agg = {
            "davies_bouldin": weighted_mean(df_metrics["davies_bouldin"].fillna(np.nan_to_num(np.nan)), weight)
            if "davies_bouldin" in df_metrics
            else np.nan,
            "silhouette": weighted_mean(df_metrics["silhouette"].fillna(np.nan_to_num(np.nan)), weight)
            if "silhouette" in df_metrics
            else np.nan,
            "calinski_harabasz": weighted_mean(df_metrics["calinski_harabasz"].fillna(np.nan_to_num(np.nan)), weight)
            if "calinski_harabasz" in df_metrics
            else np.nan,
            "total_flights": int(weight.sum()),
        }
        pd.DataFrame([agg]).to_csv(output_dir / "metrics_global.csv", index=False)

    # Save resolved config
    (output_dir / "config_resolved.yaml").write_text(yaml.dump(cfg), encoding="utf-8")
    log_path = output_dir / "experiment_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    (output_dir / "runtime_log.txt").write_text(
        json.dumps({"labels": [str(p) for p in label_paths], "experiment_log": str(log_path)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run extended clustering experiment.")
    parser.add_argument("-c", "--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--preprocessed",
        type=Path,
        default=None,
        help="Override path to preprocessed CSV. If omitted, uses config input.preprocessed_csv or latest preprocessed_*.csv in output/run/csv.",
    )
    args = parser.parse_args()
    run_experiment(args.config, preprocessed_override=args.preprocessed)
