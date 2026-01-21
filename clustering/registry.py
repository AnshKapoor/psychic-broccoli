"""Clustering registry and simple wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering,
    AffinityPropagation,
    Birch,
    DBSCAN,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    OPTICS,
    SpectralClustering,
    estimate_bandwidth,
)
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist


class Clusterer(Protocol):
    name: str
    supports_precomputed: bool

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        ...


@dataclass
class OpticsClusterer:
    name: str = "optics"
    supports_precomputed: bool = True

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        model = OPTICS(**kwargs)
        return model.fit_predict(X)


@dataclass
class DbscanClusterer:
    name: str = "dbscan"
    supports_precomputed: bool = True

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        model = DBSCAN(**kwargs)
        return model.fit_predict(X)


@dataclass
class HdbscanClusterer:
    name: str = "hdbscan"
    supports_precomputed: bool = True

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        try:
            import hdbscan  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("hdbscan is required for HDBSCAN clustering. Install via pip install hdbscan") from exc
        model = hdbscan.HDBSCAN(**kwargs)
        return model.fit_predict(X)


@dataclass
class KMeansClusterer:
    name: str = "kmeans"
    supports_precomputed: bool = False

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        model = KMeans(**kwargs)
        return model.fit_predict(X)


@dataclass
class MiniBatchKMeansClusterer:
    name: str = "minibatch_kmeans"
    supports_precomputed: bool = False

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        model = MiniBatchKMeans(**kwargs)
        return model.fit_predict(X)


@dataclass
class AgglomerativeClusterer:
    name: str = "agglomerative"
    supports_precomputed: bool = False

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        model = AgglomerativeClustering(**kwargs)
        return model.fit_predict(X)


@dataclass
class BirchClusterer:
    name: str = "birch"
    supports_precomputed: bool = False

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        model = Birch(**kwargs)
        return model.fit_predict(X)


@dataclass
class SpectralClusterer:
    name: str = "spectral"
    supports_precomputed: bool = False

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        model = SpectralClustering(**kwargs)
        return model.fit_predict(X)


@dataclass
class GmmClusterer:
    name: str = "gmm"
    supports_precomputed: bool = False
    last_model: GaussianMixture | None = None

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        if "reg_covar" in kwargs and isinstance(kwargs["reg_covar"], str):
            kwargs["reg_covar"] = float(kwargs["reg_covar"])
        model = GaussianMixture(**kwargs)
        labels = model.fit_predict(X)
        self.last_model = model
        return labels


@dataclass
class MeanShiftClusterer:
    name: str = "meanshift"
    supports_precomputed: bool = False

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        bandwidth = kwargs.pop("bandwidth", None)
        quantile = kwargs.pop("bandwidth_quantile", None)
        n_samples = kwargs.pop("bandwidth_n_samples", None)
        if bandwidth is None and quantile is not None:
            bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples)
        if bandwidth is not None:
            kwargs["bandwidth"] = bandwidth
        model = MeanShift(**kwargs)
        return model.fit_predict(X)


@dataclass
class AffinityPropagationClusterer:
    name: str = "affinity_propagation"
    supports_precomputed: bool = False

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        max_samples = kwargs.pop("max_samples", None)
        sample_random_state = kwargs.pop("sample_random_state", None)
        if max_samples is not None and X.shape[0] > max_samples:
            rng = np.random.default_rng(sample_random_state)
            idx = rng.choice(X.shape[0], size=max_samples, replace=False)
            X_sample = X[idx]
            model = AffinityPropagation(**kwargs)
            model.fit(X_sample)
            centers_idx = model.cluster_centers_indices_
            if centers_idx is None or len(centers_idx) == 0:
                labels = np.full(X.shape[0], -1, dtype=int)
                return labels
            centers = X_sample[centers_idx]
            dists = cdist(X, centers, metric="sqeuclidean")
            return np.argmin(dists, axis=1)
        model = AffinityPropagation(**kwargs)
        return model.fit_predict(X)


@dataclass
class TwoStageClusterer:
    name: str = "two_stage"
    supports_precomputed: bool = False

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        stage1_method = kwargs.pop("stage1_method", None)
        stage2_method = kwargs.pop("stage2_method", None)
        stage1_params = kwargs.pop("stage1_params", {}) or {}
        stage2_params = kwargs.pop("stage2_params", {}) or {}
        if kwargs:
            raise ValueError(f"Unsupported two_stage parameters: {sorted(kwargs)}")
        if not stage1_method or not stage2_method:
            raise ValueError("two_stage requires stage1_method and stage2_method")
        if stage1_method == "two_stage" or stage2_method == "two_stage":
            raise ValueError("two_stage cannot be nested as stage1 or stage2")

        stage1_clusterer = get_clusterer(stage1_method)
        stage1_labels = stage1_clusterer.fit_predict(X, **stage1_params)

        labels = np.full(len(stage1_labels), -1, dtype=int)
        next_label = 0
        for cluster_id in np.unique(stage1_labels):
            if cluster_id == -1:
                continue
            idx = np.where(stage1_labels == cluster_id)[0]
            if idx.size == 0:
                continue
            # Skip tiny clusters that cannot satisfy stage2 constraints.
            if idx.size < 2:
                continue

            local_params = dict(stage2_params)
            min_samples = local_params.get("min_samples")
            if min_samples is not None and min_samples > idx.size:
                local_params["min_samples"] = idx.size

            stage2_clusterer = get_clusterer(stage2_method)
            local_labels = stage2_clusterer.fit_predict(X[idx], **local_params)
            unique_local = [c for c in np.unique(local_labels) if c != -1]
            mapping = {c: next_label + i for i, c in enumerate(unique_local)}
            for pos, local_label in zip(idx, local_labels):
                labels[pos] = mapping.get(local_label, -1)
            next_label += len(unique_local)
        return labels


def get_clusterer(method_name: str) -> Clusterer:
    name = method_name.lower()
    if name == "optics":
        return OpticsClusterer()
    if name == "dbscan":
        return DbscanClusterer()
    if name == "hdbscan":
        return HdbscanClusterer()
    if name == "kmeans":
        return KMeansClusterer()
    if name in {"minibatch_kmeans", "minibatchkmeans"}:
        return MiniBatchKMeansClusterer()
    if name == "agglomerative":
        return AgglomerativeClusterer()
    if name == "birch":
        return BirchClusterer()
    if name == "spectral":
        return SpectralClusterer()
    if name in {"gmm", "gaussian_mixture"}:
        return GmmClusterer()
    if name in {"meanshift", "mean_shift"}:
        return MeanShiftClusterer()
    if name in {"affinity_propagation", "affinityprop", "affinity_propagation"}:
        return AffinityPropagationClusterer()
    if name in {"two_stage", "two-stage"}:
        return TwoStageClusterer()
    raise ValueError(f"Unsupported clustering method: {method_name}")
