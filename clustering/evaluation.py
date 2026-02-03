"""Cluster quality metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from scipy.sparse import issparse


def compute_internal_metrics(
    X_or_D,
    labels,
    metric_mode: Literal["features", "precomputed"],
    include_noise: bool = False,
) -> dict:
    labels = np.asarray(labels)
    if not include_noise:
        if metric_mode == "precomputed" and issparse(X_or_D):
            idx = np.where(labels != -1)[0]
            X_or_D = X_or_D[idx][:, idx]
            labels = labels[idx]
        else:
            mask = labels != -1
            X_or_D = X_or_D[mask]
            labels = labels[mask]

    unique = [c for c in np.unique(labels) if c != -1]
    if len(unique) < 2:
        return {
            "davies_bouldin": float("nan"),
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "n_clusters": len(unique),
            "noise_frac": float(np.mean(labels == -1)) if len(labels) else 0.0,
            "reason": "<2 clusters",
        }

    metrics = {
        "n_clusters": len(unique),
        "noise_frac": float(np.mean(labels == -1)) if include_noise else 0.0,
    }

    if metric_mode == "precomputed":
        if issparse(X_or_D):
            metrics["silhouette"] = float("nan")
            metrics["davies_bouldin"] = float("nan")
            metrics["calinski_harabasz"] = float("nan")
            metrics["reason"] = "sparse_precomputed_distances"
            return metrics
        metrics["silhouette"] = float(silhouette_score(X_or_D, labels, metric="precomputed"))
    else:
        metrics["silhouette"] = float(silhouette_score(X_or_D, labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(X_or_D, labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_or_D, labels))

    if metric_mode == "precomputed":
        metrics["davies_bouldin"] = float("nan")
        metrics["calinski_harabasz"] = float("nan")
    return metrics
