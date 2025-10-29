"""Configuration helpers for the ADS-B and noise merging pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MergerConfig:
    """Strongly-typed configuration for the ADS-B noise merger."""

    base_path: Path
    adsb_dir: Path
    excel_filename: str
    noise_csv_filename: str
    time_tolerance_min: float
    log_level: int


_ACTIVE_CONFIG: Optional[MergerConfig] = None


def set_active_config(config: MergerConfig) -> None:
    """Register a configuration instance for reuse across helper functions."""

    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config


def get_active_config() -> Optional[MergerConfig]:
    """Return the currently active configuration if one has been registered."""

    return _ACTIVE_CONFIG


def resolve_config(base_path: str) -> MergerConfig:
    """Build a default :class:`MergerConfig` for the supplied base path."""

    base_path_obj: Path = Path(base_path).resolve()
    active_config: Optional[MergerConfig] = get_active_config()

    if active_config and active_config.base_path == base_path_obj:
        return active_config

    return MergerConfig(
        base_path=base_path_obj,
        adsb_dir=base_path_obj / "adsb",
        excel_filename="noise_measurements.xlsx",
        noise_csv_filename="combined_noise_measurements.csv",
        time_tolerance_min=10.0,
        log_level=logging.INFO,
    )
