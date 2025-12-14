"""Experiment runner for extended clustering and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from clustering.distances import build_feature_matrix, pairwise_distance_matrix
from clustering.evaluation import compute_internal_metrics
from clustering.registry import get_clusterer


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def filter_flows(df: pd.DataFrame, flow_keys: List[str], include: List[List[str]] | None) -> pd.DataFrame:
    if not include:
        return df
    include_set = {tuple(item) for item in include}
    mask = df[flow_keys].apply(tuple, axis=1).isin(include_set)
    return df[mask].reset_index(drop=True)


def weighted_mean(values: List[float], weights: List[int]) -> float:
    vals = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    return float(np.average(vals, weights=w))


def run_experiment(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    clustering_cfg: Dict[str, object] = cfg.get("clustering", {}) or {}
    eval_cfg: Dict[str, object] = clustering_cfg.get("evaluation", {}) or {}
    method = clustering_cfg.get("method", "optics")
    distance_metric = clustering_cfg.get("distance_metric", "euclidean")
    experiment_name = cfg.get("output", {}).get("experiment_name", "experiment")

    flows_cfg = cfg.get("flows", {}) or {}
    flow_keys = flows_cfg.get("flow_keys", ["Runway"])
    include_flows = flows_cfg.get("include", []) or []

    input_cfg = cfg.get("input", {}) or {}
    preprocessed_csv = input_cfg.get("preprocessed_csv", "output/run/csv/preprocessed.csv")
    df = pd.read_csv(preprocessed_csv)
    df = filter_flows(df, flow_keys, include_flows)

    output_dir = Path(cfg.get("output", {}).get("dir", "output")) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare outputs
    metrics_rows: List[dict] = []
    label_paths: List[Path] = []

    # Per-flow clustering
    for flow_vals, flow_df in df.groupby(flow_keys):
        if not isinstance(flow_vals, tuple):
            flow_vals = (flow_vals,)
        flow_name = "_".join(str(v) for v in flow_vals)

        # Build features/trajectories
        vector_cols = cfg.get("features", {}).get("vector_cols", ["x_utm", "y_utm"])
        X, trajs = build_feature_matrix(flow_df, vector_cols=vector_cols)

        clusterer = get_clusterer(method)
        precomputed_needed = distance_metric in {"dtw", "frechet"}

        if precomputed_needed:
            D = pairwise_distance_matrix(
                trajs if distance_metric in {"dtw", "frechet"} else X,
                metric=distance_metric,
                cache_dir=output_dir / "cache",
                flow_name=flow_name,
                params={
                    "distance_metric": distance_metric,
                    "n_points": cfg.get("preprocessing", {}).get("resampling", {}).get("n_points"),
                },
            )
            labels = clusterer.fit_predict(D, metric="precomputed")
            metrics = compute_internal_metrics(D, labels, metric_mode="precomputed", include_noise=eval_cfg.get("include_noise", False))
        else:
            labels = clusterer.fit_predict(
                X,
                **(clustering_cfg.get(method, {}) or {}),
            )
            metrics = compute_internal_metrics(X, labels, metric_mode="features", include_noise=eval_cfg.get("include_noise", False))

        metrics.update({key: val for key, val in zip(flow_keys, flow_vals)})
        metrics["n_flights"] = len(labels)
        metrics_rows.append(metrics)

        labeled = flow_df.drop_duplicates(subset=["flight_id"]).copy()
        labeled["cluster_id"] = labels
        label_path = output_dir / f"labels_{flow_name}.parquet"
        labeled.to_parquet(label_path, index=False)
        label_paths.append(label_path)

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
    (output_dir / "runtime_log.txt").write_text(
        json.dumps({"labels": [str(p) for p in label_paths]}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run extended clustering experiment.")
    parser.add_argument("-c", "--config", type=Path, required=True, help="Path to YAML config.")
    args = parser.parse_args()
    run_experiment(args.config)
