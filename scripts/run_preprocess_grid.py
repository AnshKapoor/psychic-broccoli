"""Run a preprocessing grid to generate standardized preprocessed_*.csv files.

Usage:
  python scripts/run_preprocess_grid.py --grid config/preprocess_grid.yaml
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.save_preprocessed import main as save_preprocessed_main


def _deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def main(grid_path: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    grid = yaml.safe_load(grid_path.read_text(encoding="utf-8")) or {}
    base_cfg_path = Path(grid.get("base_config", "config/backbone_full.yaml"))
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8")) or {}

    variants = grid.get("variants", [])
    if not variants:
        raise ValueError("No variants found in preprocess grid.")

    output_dir = Path(base_cfg.get("output", {}).get("dir", "data")) / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, variant in enumerate(variants, start=1):
        cfg = copy.deepcopy(base_cfg)
        cfg = _deep_update(cfg, variant)
        if "output" not in cfg:
            cfg["output"] = {}
        if "preprocessed_id" in variant:
            cfg["output"]["preprocessed_id"] = variant["preprocessed_id"]
        pre_id = cfg["output"].get("preprocessed_id")
        if pre_id is not None:
            out_path = output_dir / f"preprocessed_{int(pre_id)}.csv"
            if out_path.exists():
                logging.info("Skip %s (already exists)", out_path)
                continue

        tmp_path = Path(f".tmp_preprocess_cfg_{idx}.yaml")
        tmp_path.write_text(yaml.dump(cfg), encoding="utf-8")
        try:
            save_preprocessed_main(str(tmp_path))
            if pre_id is not None:
                logging.info("Saved preprocessed_%s.csv", int(pre_id))
        except Exception as exc:
            logging.error("Preprocess variant %s failed: %s", idx, exc)
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing grid.")
    parser.add_argument("--grid", type=Path, default=Path("config/preprocess_grid.yaml"))
    args = parser.parse_args()
    main(args.grid)
