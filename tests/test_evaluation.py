import numpy as np

from clustering.evaluation import compute_internal_metrics


def test_all_noise_returns_nan():
    X = np.random.rand(5, 2)
    labels = np.array([-1, -1, -1, -1, -1])
    metrics = compute_internal_metrics(X, labels, metric_mode="features", include_noise=False)
    assert np.isnan(metrics["silhouette"])
    assert metrics["reason"] == "<2 clusters"


def test_single_cluster_returns_nan():
    X = np.random.rand(5, 2)
    labels = np.array([0, 0, 0, 0, 0])
    metrics = compute_internal_metrics(X, labels, metric_mode="features", include_noise=False)
    assert np.isnan(metrics["silhouette"])
