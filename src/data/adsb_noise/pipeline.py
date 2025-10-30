"""High-level pipeline orchestration for the ADS-B noise merger."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import MergerConfig
from .exporter import MergedDatasetExporter
from .grouping import FlightGrouper
from .loader import ADSBBatchLoader, NoiseWorkbookLoader
from .matcher import FlightNoiseMatcher
from .preprocessor import FlightPreprocessor


class ADSNoiseMerger:
    """Coordinate the end-to-end workflow for merging ADS-B and noise data."""

    def __init__(self, config: MergerConfig) -> None:
        """Initialise helpers using the provided configuration."""

        self.config: MergerConfig = config
        self.adsb_loader = ADSBBatchLoader(config)
        self.noise_loader = NoiseWorkbookLoader(config)
        self.grouper = FlightGrouper(config)
        self.preprocessor = FlightPreprocessor(config)
        self.matcher = FlightNoiseMatcher(config)
        self.exporter = MergedDatasetExporter(config)

    def run(self) -> Tuple[pd.DataFrame, Path]:
        """Execute the complete pipeline and return merged data with CSV path."""

        adsb_data: pd.DataFrame = self.adsb_loader.load()
        flights: pd.DataFrame = self.grouper.group(adsb_data)
        processed_flights: pd.DataFrame = self.preprocessor.preprocess(flights)
        noise_data: pd.DataFrame = self.noise_loader.load()
        merged: pd.DataFrame = self.matcher.match(processed_flights, noise_data)
        # Exporting happens last to keep intermediate frames available for inspection.
        merged, csv_path = self.exporter.export(merged)
        return merged, csv_path
