"""Heuristics for inferring flight metadata used during matching."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def infer_ad_and_runway_for_flight(flight_df: pd.DataFrame) -> Dict[str, object]:
    """Infer arrival/departure flag and runway timestamp from a trajectory."""

    altitude_series: pd.Series = flight_df.get("altitude", pd.Series(dtype=float)).dropna()
    timestamps: pd.Series = pd.to_datetime(flight_df.get("timestamp"), utc=True, errors="coerce").dropna()

    if not altitude_series.empty:
        ad_flag: str = "A" if altitude_series.iloc[-1] < altitude_series.iloc[0] else "D"
    else:
        ad_flag = "D"

    if not timestamps.empty:
        runway_time: pd.Timestamp = (
            timestamps.iloc[-1] if ad_flag == "A" else timestamps.iloc[0]
        )
    else:
        runway_time = pd.NaT

    return {"A/D": ad_flag, "Runway": None, "runway_time": runway_time}
