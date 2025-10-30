"""Flight grouping utilities for ADS-B trajectories."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .config import MergerConfig
from .base import PipelineComponent


class FlightGrouper(PipelineComponent):
    """Aggregate ADS-B points into per-flight trajectories."""

    def __init__(self, config: MergerConfig) -> None:
        """Store configuration for future grouping operations."""

        super().__init__(config)

    def group(self, adsb_data: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame that stores grouped trajectories as dictionaries."""

        required_columns = {"icao24", "callsign", "day"}
        if not required_columns.issubset(adsb_data.columns):
            available_columns: List[str] = sorted(map(str, adsb_data.columns.tolist()))
            message: str = "ADS-B data must contain 'icao24', 'callsign', and 'day' columns."
            self.logger.error("%s Available columns: %s", message, available_columns)
            # Raising the KeyError preserves the existing control flow while
            # providing the caller with the information necessary to diagnose
            # unexpected data schemas quickly.
            raise KeyError(message)

        grouped_records: List[Dict[str, object]] = []
        for (icao24, callsign, day), group in adsb_data.groupby(["icao24", "callsign", "day"], dropna=True):
            group_sorted: pd.DataFrame = group.sort_values("timestamp").reset_index(drop=True)
            if len(group_sorted) < 3:
                continue
            grouped_records.append(
                {
                    "icao24": icao24,
                    "callsign": callsign,
                    "day": day,
                    # Store the cleaned and time-ordered trajectory for later steps.
                    "trajectory": group_sorted,
                }
            )

        flights_df: pd.DataFrame = pd.DataFrame(grouped_records)
        self.logger.info("Constructed %d flight trajectories after grouping.", len(flights_df))
        return flights_df
