"""Data loading helpers for the ADS-B noise pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import joblib
import pandas as pd

from src.data.data_preparation import combine_noise_data_sheets, match_flight_bp_noise

from .config import MergerConfig
from .base import PipelineComponent


class ADSBBatchLoader(PipelineComponent):
    """Load and clean monthly ADS-B batches stored as ``.joblib`` files."""

    def __init__(self, config: MergerConfig) -> None:
        """Store configuration for later reuse while loading files."""

        super().__init__(config)

    def load(self) -> pd.DataFrame:
        """Return a concatenated ADS-B :class:`~pandas.DataFrame` assembled from ``.joblib`` batches.

        The loader gracefully tolerates missing identifier columns by only
        sorting with the fields that are present, ensuring that data pulled from
        heterogeneous sources can still be processed.
        """

        adsb_dir: Path = self.config.adsb_dir
        joblib_files: List[Path] = sorted(adsb_dir.glob("*.joblib"))

        if not joblib_files:
            message: str = f"No .joblib files found in {adsb_dir}."
            self.logger.error(message)
            raise FileNotFoundError(message)

        frames: List[pd.DataFrame] = []
        for joblib_file in joblib_files:
            try:
                data = joblib.load(joblib_file)
            except Exception as exc:  # pragma: no cover - I/O error handling
                self.logger.warning("Failed to load %s: %s", joblib_file, exc)
                continue

            if isinstance(data, pd.DataFrame):
                frame: pd.DataFrame = data.copy()
            elif isinstance(data, Iterable):
                frame = pd.DataFrame(list(data))
            else:
                self.logger.warning("Unsupported data structure in %s", joblib_file)
                continue

            # Persist each successfully loaded frame for concatenation later.
            frames.append(frame)

        if not frames:
            message = "ADS-B directory did not yield any usable DataFrame."
            self.logger.error(message)
            raise ValueError(message)

        adsb_data: pd.DataFrame = pd.concat(frames, ignore_index=True)

        for column in ("timestamp", "firstseen", "lastseen"):
            if column in adsb_data.columns:
                adsb_data[column] = pd.to_datetime(adsb_data[column], utc=True, errors="coerce")

        if "callsign" in adsb_data.columns:
            adsb_data["callsign"] = (
                adsb_data["callsign"].astype(str).str.upper().str.strip().replace({"nan": pd.NA})
            )

        # Ensure spatial and temporal coordinates exist before filtering rows with missing values.
        required_columns: List[str] = ["timestamp", "latitude", "longitude"]
        existing_required_columns: List[str] = [col for col in required_columns if col in adsb_data.columns]
        if existing_required_columns:
            adsb_data = adsb_data.dropna(subset=existing_required_columns)
        missing_required_columns: List[str] = [
            column_name for column_name in required_columns if column_name not in adsb_data.columns
        ]
        if missing_required_columns:
            self.logger.warning(
                "ADS-B dataset missing required columns for validation: %s", missing_required_columns
            )

        # Sort the ADS-B points by the identifiers that are available to keep the
        # chronological order stable even when optional columns are missing.
        sort_priority_columns: List[str] = [
            column_name
            for column_name in ("icao24", "callsign", "timestamp")
            if column_name in adsb_data.columns
        ]
        if sort_priority_columns:
            adsb_data = adsb_data.sort_values(by=sort_priority_columns, na_position="last")
        else:
            self.logger.warning(
                "ADS-B dataset missing expected identifier columns; skipping sort. Available columns: %s",
                list(adsb_data.columns),
            )

        if "day" in adsb_data.columns:
            adsb_data["day"] = pd.to_datetime(adsb_data["day"], errors="coerce").dt.date

        self.logger.info("Loaded %d ADS-B points from %d files.", len(adsb_data), len(frames))
        return adsb_data

    def list_columns(self) -> List[str]:
        """Return a sorted list of column names present in the ADS-B dataset.

        Returns:
            A list containing the unique column names discovered after loading
            the ADS-B batches. The names are sorted alphabetically so they are
            easy to scan when printed on the command line.
        """

        # Reuse the existing load routine so the column listing reflects every
        # transformation and normalisation step applied by the loader.
        adsb_data: pd.DataFrame = self.load()
        column_names: List[str] = sorted(dict.fromkeys(map(str, adsb_data.columns.tolist())))
        self.logger.info("ADS-B dataset exposes %d columns.", len(column_names))
        return column_names


class NoiseWorkbookLoader(PipelineComponent):
    """Load and normalise the noise workbook prior to matching."""

    def __init__(self, config: MergerConfig) -> None:
        """Store configuration for workbook ingestion."""

        super().__init__(config)

    def load(self) -> pd.DataFrame:
        """Return a combined noise :class:`~pandas.DataFrame` from Excel sheets."""

        noise_df: pd.DataFrame = combine_noise_data_sheets(
            str(self.config.base_path), self.config.excel_filename, self.config.noise_csv_filename
        )

        if noise_df.empty:
            self.logger.warning("Noise workbook produced an empty dataset.")

        if "ATA/ATD" in noise_df.columns:
            noise_df["ATA/ATD"] = pd.to_datetime(noise_df["ATA/ATD"], utc=True, errors="coerce")
            noise_df["day"] = noise_df["ATA/ATD"].dt.date

        for column in ("A/D", "Runway"):
            if column not in noise_df.columns:
                noise_df[column] = pd.NA

        noise_df["A/D"] = noise_df["A/D"].astype(str).str.upper().str.strip().replace({"nan": pd.NA})
        noise_df["Runway"] = (
            noise_df["Runway"].astype(str).str.upper().str.strip().replace({"nan": pd.NA, "": pd.NA})
        )

        self._invoke_legacy_matcher(noise_df)

        self.logger.info("Loaded %d noise measurements from workbook.", len(noise_df))
        return noise_df

    def _invoke_legacy_matcher(self, noise_df: pd.DataFrame) -> None:
        """Invoke the legacy matcher to honour historical side effects."""

        try:
            dummy_bp_data = pd.DataFrame(columns=noise_df.columns)
            match_flight_bp_noise(dummy_bp_data, noise_df, str(self.config.base_path))
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.debug("Legacy matcher invocation failed harmlessly: %s", exc)
