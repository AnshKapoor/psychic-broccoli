"""Output helpers for writing merged ADS-B and noise datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import pandas as pd

from .config import MergerConfig
from .base import PipelineComponent


class MergedDatasetExporter(PipelineComponent):
    """Write merged datasets to disk in CSV and Parquet formats."""

    def __init__(self, config: MergerConfig) -> None:
        """Store configuration for locating the output directory."""

        super().__init__(config)

    @staticmethod
    def _json_default(value: Any) -> str:
        """Convert unsupported JSON types to readable strings.

        The exporter needs to serialise nested trajectory points that contain
        :class:`pandas.Timestamp` objects and ``numpy`` scalar dtypes.  Falling
        back to ``str`` ensures that every element can be converted to JSON
        without raising ``TypeError``.
        """

        return str(value)

    @classmethod
    def _serialise_nested(cls, value: Any) -> str:
        """Return a JSON string for complex nested structures.

        ``json.dumps`` receives :meth:`_json_default` as ``default`` to make sure
        timestamps and numpy dtypes are represented as strings instead of
        crashing the export.
        """

        return json.dumps(value, default=cls._json_default)

    def export(self, merged: pd.DataFrame) -> Tuple[pd.DataFrame, Path]:
        """Persist the merged dataset and return the saved CSV path."""

        output_dir: Path = self.config.base_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path: Path = output_dir / "merged_noise_flights.csv"
        parquet_path: Path = output_dir / "merged_noise_flights.parquet"

        serialised: pd.DataFrame = merged.copy()
        if "raw_points" in serialised.columns:
            # Use safe JSON dumping so timestamps and numpy values become strings.
            serialised["raw_points"] = serialised["raw_points"].apply(
                self._serialise_nested
            )
        if "processed_points" in serialised.columns:
            # Apply the same conversion for processed trajectories to avoid crashes.
            serialised["processed_points"] = serialised["processed_points"].apply(
                self._serialise_nested
            )

        # Write the serialised representation so nested structures survive round-trips.
        serialised.to_csv(csv_path, index=False)

        try:
            merged.to_parquet(parquet_path, index=False)
        except Exception as exc:  # pragma: no cover - optional dependency
            self.logger.warning("Writing Parquet failed (%s). Continuing without it.", exc)

        self.logger.info("Merged dataset written to %s and %s.", csv_path, parquet_path)
        return merged, csv_path
