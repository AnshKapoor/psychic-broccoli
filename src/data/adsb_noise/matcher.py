"""Matching helpers that align flights with noise measurements."""

from __future__ import annotations

from typing import List

import pandas as pd

from .config import MergerConfig
from .base import PipelineComponent


class FlightNoiseMatcher(PipelineComponent):
    """Perform time-tolerant matching between flights and noise rows."""

    def __init__(self, config: MergerConfig) -> None:
        """Store configuration for matching operations."""

        super().__init__(config)

    def match(self, flights: pd.DataFrame, noise_data: pd.DataFrame) -> pd.DataFrame:
        """Return flight/noise matches with time deltas after enforcing sorted groupings."""

        if flights.empty:
            self.logger.warning("No flights available for matching.")
            return pd.DataFrame()

        tolerance = pd.Timedelta(minutes=self.config.time_tolerance_min)

        flights = flights.copy()
        noise_data = noise_data.copy()

        flights["day"] = pd.to_datetime(flights["day"], errors="coerce").dt.date
        noise_data["day"] = pd.to_datetime(noise_data.get("day"), errors="coerce").dt.date

        flights = flights[flights["runway_time"].notna()]
        noise_data = noise_data.sort_values("ATA/ATD")

        merged_frames: List[pd.DataFrame] = []

        runway_flights: pd.DataFrame = flights[flights["Runway"].notna()]
        runway_noise: pd.DataFrame = noise_data[noise_data["Runway"].notna()]

        if not runway_flights.empty:
            # Sort by the merge key and group-by columns to keep merge_asof deterministic.
            runway_flights = runway_flights.sort_values(["A/D", "Runway", "day", "runway_time"])
        if not runway_noise.empty:
            # Matching on runway also requires the noise rows to respect the same ordering.
            runway_noise = runway_noise.sort_values(["A/D", "Runway", "day", "ATA/ATD"])

        if not runway_flights.empty and not runway_noise.empty:
            # Prioritise matches where both sources agree on the runway identifier.
            runway_match = pd.merge_asof(
                runway_flights,
                runway_noise,
                left_on="runway_time",
                right_on="ATA/ATD",
                by=["A/D", "Runway", "day"],
                tolerance=tolerance,
                direction="nearest",
                suffixes=("_flight", "_noise"),
            )
            merged_frames.append(runway_match)

        fallback_flights: pd.DataFrame = flights[flights["Runway"].isna()]
        fallback_noise: pd.DataFrame = noise_data.copy()

        if not fallback_flights.empty:
            # Sorting ensures merge_asof respects the A/D-day partitions even without runway data.
            fallback_flights = fallback_flights.sort_values(["A/D", "day", "runway_time"])
        if not fallback_noise.empty:
            # The fallback noise rows need the identical ordering guarantees.
            fallback_noise = fallback_noise.sort_values(["A/D", "day", "ATA/ATD"])
        if not fallback_flights.empty:
            # Fallback to matching solely on arrival/departure direction when runway is missing.
            fallback_match = pd.merge_asof(
                fallback_flights,
                fallback_noise,
                left_on="runway_time",
                right_on="ATA/ATD",
                by=["A/D", "day"],
                tolerance=tolerance,
                direction="nearest",
                suffixes=("_flight", "_noise"),
            )
            merged_frames.append(fallback_match)

        if not merged_frames:
            self.logger.warning("No matches found between flights and noise data.")
            return flights

        merged = pd.concat(merged_frames, ignore_index=True, sort=False)

        if "ATA/ATD" in merged.columns:
            merged["time_delta_s"] = (merged["ATA/ATD"] - merged["runway_time"]).dt.total_seconds().abs()
        else:
            merged["time_delta_s"] = pd.NA

        merged["matched"] = merged["ATA/ATD"].notna()

        self.logger.info("Matched %d flights with noise measurements.", merged["matched"].sum())
        return merged
