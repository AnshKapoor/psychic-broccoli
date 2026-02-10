"""Run common EDA tasks (raw lengths + flight counts) in one step.

Usage:
  python scripts/run_eda_combo.py -c config/backbone_full.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eda_flight_counts import main as flight_counts_main
from scripts.eda_raw_lengths import main as raw_lengths_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run common EDA tasks.")
    parser.add_argument("-c", "--config", default="config/backbone_full.yaml")
    args = parser.parse_args()

    # Ensure both scripts use the same config by injecting argv.
    sys.argv = ["eda_raw_lengths.py", "-c", args.config, "--outdir", "output/eda/raw_lengths"]
    raw_lengths_main()
    sys.argv = ["eda_flight_counts.py", "-c", args.config, "--outdir", "output/eda/flight_counts"]
    flight_counts_main()


if __name__ == "__main__":
    main()
