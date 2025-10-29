"""Output helpers for writing merged ADS-B and noise datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import MergerConfig
from .base import PipelineComponent


class MergedDatasetExporter(PipelineComponent):
    """Write merged datasets to disk in CSV and Parquet formats."""

    def __init__(self, config: MergerConfig) -> None:
        """Store configuration for locating the output directory."""

        super().__init__(config)

    def export(self, merged: pd.DataFrame) -> Tuple[pd.DataFrame, Path]:
        """Persist the merged dataset and return the saved CSV path."""

        output_dir: Path = self.config.base_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path: Path = output_dir / "merged_noise_flights.csv"
        parquet_path: Path = output_dir / "merged_noise_flights.parquet"

        serialised = merged.copy()
        if "raw_points" in serialised.columns:
            serialised["raw_points"] = serialised["raw_points"].apply(json.dumps)
        if "processed_points" in serialised.columns:
            serialised["processed_points"] = serialised["processed_points"].apply(json.dumps)

        # Write the serialised representation so nested structures survive round-trips.
        serialised.to_csv(csv_path, index=False)

        try:
            merged.to_parquet(parquet_path, index=False)
        except Exception as exc:  # pragma: no cover - optional dependency
            self.logger.warning("Writing Parquet failed (%s). Continuing without it.", exc)

        self.logger.info("Merged dataset written to %s and %s.", csv_path, parquet_path)
        return merged, csv_path
