# %% [markdown]
"""Command-line entry point for merging ADS-B trajectories with noise data."""

# %% Imports
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

try:
    # Attempt the standard package import when the project root is on ``sys.path``.
    from src.data.adsb_noise import ADSNoiseMerger, MergerConfig, resolve_config, set_active_config
except ModuleNotFoundError as import_error:
    # Support executing the file directly (``python src/data/adsb_noise_merge.py``) by
    # injecting the repository root into ``sys.path`` before re-importing the package.
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from src.data.adsb_noise import ADSNoiseMerger, MergerConfig, resolve_config, set_active_config

    # Re-raise the original error if the import still fails to give clear diagnostics.
    if "src" not in sys.modules:
        raise import_error

# %% Public orchestrator

def prepare_merged_dataset(base_path: str) -> Tuple[pd.DataFrame, str]:
    """Return the merged dataset together with the CSV export path."""

    config: MergerConfig = resolve_config(base_path)
    # Ensure logging honours the configuration before running the pipeline.
    logging.getLogger().setLevel(config.log_level)
    merger = ADSNoiseMerger(config)
    merged_df, csv_path = merger.run()
    return merged_df, str(csv_path)


# %% CLI helpers

def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments exposed by the merger utility."""

    parser = argparse.ArgumentParser(description="Merge ADS-B trajectories with noise measurements.")
    parser.add_argument("--base-path", required=True, help="Base directory containing ADS-B and noise data.")
    parser.add_argument(
        "--adsb-dir",
        default=None,
        help="Directory with monthly ADS-B .joblib files (defaults to <base-path>/adsb).",
    )
    parser.add_argument(
        "--excel",
        default="noise_measurements.xlsx",
        help="Excel workbook containing noise measurements.",
    )
    parser.add_argument(
        "--time-tolerance-min",
        type=float,
        default=10.0,
        help="Time tolerance in minutes for matching flights with noise rows.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level for the script's output.",
    )
    return parser.parse_args(args=args)


# %% CLI entry point

def main(cli_args: Optional[Iterable[str]] = None) -> None:
    """Execute the merge pipeline using command-line arguments."""

    args = parse_args(cli_args)

    base_path = Path(args.base_path).resolve()
    # Determine where ADS-B files reside, defaulting to the conventional folder.
    adsb_dir = Path(args.adsb_dir).resolve() if args.adsb_dir else base_path / "adsb"

    # Configure logging using the requested verbosity for consistent diagnostics.
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    config = MergerConfig(
        base_path=base_path,
        adsb_dir=adsb_dir,
        excel_filename=args.excel,
        noise_csv_filename="combined_noise_measurements.csv",
        time_tolerance_min=args.time_tolerance_min,
        log_level=log_level,
    )
    set_active_config(config)

    merged_df, csv_path = prepare_merged_dataset(str(base_path))
    logging.info("Merged %d records. CSV saved to %s", len(merged_df), csv_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
