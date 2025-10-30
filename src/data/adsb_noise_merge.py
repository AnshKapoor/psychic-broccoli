# %% [markdown]
"""Command-line entry point for merging ADS-B trajectories with noise data."""

# %% Imports
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd

try:
    # Attempt the standard package import when the project root is on ``sys.path``.
    from src.data.adsb_noise import ADSNoiseMerger, MergerConfig, resolve_config, set_active_config
    from src.data.adsb_noise.loader import ADSBBatchLoader
except ModuleNotFoundError as import_error:
    # Support executing the file directly (``python src/data/adsb_noise_merge.py``) by
    # injecting the repository root into ``sys.path`` before re-importing the package.
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from src.data.adsb_noise import ADSNoiseMerger, MergerConfig, resolve_config, set_active_config
    from src.data.adsb_noise.loader import ADSBBatchLoader

    # Re-raise the original error if the import still fails to give clear diagnostics.
    if "src" not in sys.modules:
        raise import_error

# %% Public orchestrator

def prepare_merged_dataset(base_path: Optional[Union[str, Path]]) -> Tuple[pd.DataFrame, str]:
    """Return the merged dataset together with the CSV export path.

    Args:
        base_path: Optional string or :class:`pathlib.Path` pointing to the working directory that contains
            ADS-B and noise measurement artifacts. When ``None`` or an empty
            string is supplied, the current working directory is used.

    Returns:
        A tuple containing the merged :class:`pandas.DataFrame` result together
        with the CSV export path as a string to simplify downstream logging.
    """

    # Normalise the base path so that ``None`` or an empty string fall back to
    # the current working directory, mirroring the CLI behaviour.
    fallback_base_path: Union[str, Path] = base_path or Path(".")
    resolved_base_path: str = str(fallback_base_path)
    config: MergerConfig = resolve_config(resolved_base_path)
    # Ensure logging honours the configuration before running the pipeline.
    logging.getLogger().setLevel(config.log_level)
    merger: ADSNoiseMerger = ADSNoiseMerger(config)
    merged_df, csv_path = merger.run()
    return merged_df, str(csv_path)


# %% CLI helpers

def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments exposed by the merger utility."""

    parser = argparse.ArgumentParser(description="Merge ADS-B trajectories with noise measurements.")
    parser.add_argument(
        "--base-path",
        default=None,
        help="Base directory containing ADS-B and noise data (defaults to the current working directory).",
    )
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
    parser.add_argument(
        "--list-adsb-columns",
        action="store_true",
        help="Display all ADS-B column names after loading the dataset and exit.",
    )
    return parser.parse_args(args=args)


# %% CLI entry point

def show_adsb_columns(config: MergerConfig) -> None:
    """Print the ADS-B column names discovered via the configured loader."""

    loader: ADSBBatchLoader = ADSBBatchLoader(config)
    column_names: List[str] = loader.list_columns()
    # Emit a friendly, sorted list to stdout so the CLI can be used as a quick
    # inspection tool prior to running the full pipeline.
    for column_name in column_names:
        print(column_name)


def main(cli_args: Optional[Iterable[str]] = None) -> None:
    """Execute the merge pipeline using command-line arguments."""

    args = parse_args(cli_args)

    # Resolve the base path, accepting ``None`` or an empty string as a request
    # to rely on the current working directory to simplify CLI usage.
    base_path_input: Optional[str] = args.base_path
    base_path: Path = Path(base_path_input or ".").resolve()
    # Determine where ADS-B files reside, defaulting to the conventional folder.
    adsb_dir: Path = Path(args.adsb_dir).resolve() if args.adsb_dir else base_path / "adsb"

    # Configure logging using the requested verbosity for consistent diagnostics.
    log_level: int = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    config: MergerConfig = MergerConfig(
        base_path=base_path,
        adsb_dir=adsb_dir,
        excel_filename=args.excel,
        noise_csv_filename="combined_noise_measurements.csv",
        time_tolerance_min=args.time_tolerance_min,
        log_level=log_level,
    )
    set_active_config(config)

    if args.list_adsb_columns:
        show_adsb_columns(config)
        return

    merged_df: pd.DataFrame
    csv_path: str
    merged_df, csv_path = prepare_merged_dataset(str(base_path))
    logging.info("Merged %d records. CSV saved to %s", len(merged_df), csv_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
