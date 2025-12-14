"""Clustering registry and simple wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, OPTICS


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
class AgglomerativeClusterer:
    name: str = "agglomerative"
    supports_precomputed: bool = False

    def fit_predict(self, X, **kwargs) -> np.ndarray:
        model = AgglomerativeClustering(**kwargs)
        return model.fit_predict(X)


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
    if name == "agglomerative":
        return AgglomerativeClusterer()
    raise ValueError(f"Unsupported clustering method: {method_name}")
