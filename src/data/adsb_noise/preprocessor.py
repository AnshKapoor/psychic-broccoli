"""Flight preprocessing helpers that reuse project-wide utilities."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.data.trajectory_preprocessing import preprocess_trajectory

from .config import MergerConfig
from .heuristics import infer_ad_and_runway_for_flight
from .base import PipelineComponent


class FlightPreprocessor(PipelineComponent):
    """Preprocess grouped flights and enrich them with metadata."""

    def __init__(self, config: MergerConfig) -> None:
        """Store configuration for later preprocessing steps."""

        super().__init__(config)

    def preprocess(self, flights: pd.DataFrame) -> pd.DataFrame:
        """Return enriched flights containing raw and preprocessed trajectories."""

        processed_records: List[Dict[str, object]] = []

        for _, row in flights.iterrows():
            trajectory: pd.DataFrame = row["trajectory"]
            preprocessed: pd.DataFrame = preprocess_trajectory(trajectory, target_length=20)
            metadata: Dict[str, object] = infer_ad_and_runway_for_flight(trajectory)

            record: Dict[str, object] = {
                "icao24": row["icao24"],
                "callsign": row["callsign"],
                "day": row["day"],
                "A/D": metadata.get("A/D"),
                "Runway": metadata.get("Runway"),
                # Capture the inferred runway timestamp used during matching.
                "runway_time": metadata.get("runway_time"),
                "raw_points": trajectory.to_dict(orient="records"),
                "processed_points": preprocessed.to_dict(orient="records") if not preprocessed.empty else [],
            }

            processed_records.append(record)

        processed_df: pd.DataFrame = pd.DataFrame(processed_records)
        processed_df["runway_time"] = pd.to_datetime(processed_df["runway_time"], utc=True, errors="coerce")
        self.logger.info("Preprocessed %d flights.", len(processed_df))
        return processed_df
