"""Backbone computation from clustered trajectories.

Computes per-step percentile envelopes (e.g., p10/p50/p90) for each clustered
flow, yielding backbone and side tracks with flight counts. Supports UTM-based
coordinates when available.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def compute_backbones(
    df: pd.DataFrame,
    percentiles: List[int],
    min_flights_per_cluster: int,
    use_utm: bool = False,
) -> pd.DataFrame:
    """
    For each (A/D, Runway, cluster_id), compute per-step percentiles for trajectory variables.
    """

    if df.empty or "cluster_id" not in df.columns:
        return pd.DataFrame()

    rows = []
    for (ad, runway, cluster_id), cluster_df in df.groupby(["A/D", "Runway", "cluster_id"]):
        n_flights = cluster_df["flight_id"].nunique()
        if n_flights < min_flights_per_cluster or cluster_id == -1:
            continue

        for step, step_df in cluster_df.groupby("step"):
            for p in percentiles:
                rows.append(
                    {
                        "A/D": ad,
                        "Runway": runway,
                        "cluster_id": cluster_id,
                        "step": step,
                        "percentile": p,
                        "x_utm": np.percentile(step_df["x_utm"], p) if use_utm and "x_utm" in step_df else None,
                        "y_utm": np.percentile(step_df["y_utm"], p) if use_utm and "y_utm" in step_df else None,
                        "latitude": np.percentile(step_df["latitude"], p) if "latitude" in step_df else None,
                        "longitude": np.percentile(step_df["longitude"], p) if "longitude" in step_df else None,
                        "altitude": np.percentile(step_df["altitude"], p)
                        if "altitude" in step_df
                        else None,
                        "dist_to_airport_m": np.percentile(step_df["dist_to_airport_m"], p)
                        if "dist_to_airport_m" in step_df
                        else None,
                        "n_flights": n_flights,
                    }
                )

    return pd.DataFrame(rows)
