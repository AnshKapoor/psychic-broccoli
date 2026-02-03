"""Run a grid of clustering experiments defined in a YAML file."""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runner import run_experiment


def iter_param_grid(param_dict: dict) -> list[dict]:
    keys = []
    values = []
    for key, val in param_dict.items():
        if isinstance(val, list):
            keys.append(key)
            values.append(val)
    if not keys:
        return [param_dict]
    combos = []
    for product_vals in itertools.product(*values):
        cfg = copy.deepcopy(param_dict)
        for k, v in zip(keys, product_vals):
            cfg[k] = v
        combos.append(cfg)
    return combos


def main(grid_path: Path) -> None:
    grid = yaml.safe_load(grid_path.read_text(encoding="utf-8")) or {}
    experiments = grid.get("experiments", [])
    base_output = grid.get("output", {}).get("dir", "output")
    base_cfg_path = Path("config/backbone_full.yaml")
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))

    for exp in experiments:
        exp_name = exp["name"]
        method = exp["method"]
        distance_metric = exp.get("distance_metric", "euclidean")

        # Prepare parameter sweeps
        param_space = []
        method_params = exp.get(method, {})
        if method_params:
            param_space = iter_param_grid(method_params)
        else:
            param_space = [{}]

        explicit_name = exp.get("experiment_name")
        if len(param_space) > 1 and not explicit_name:
            raise ValueError(
                f"Experiment '{exp_name}' expands to multiple runs but has no explicit experiment_name. "
                "Provide explicit experiment_name entries for each run to keep numbering stable."
            )

        for idx, params in enumerate(param_space, start=1):
            cfg = copy.deepcopy(base_cfg)
            cfg["clustering"]["method"] = method
            cfg["clustering"]["distance_metric"] = distance_metric
            cfg["clustering"][method] = params
            grid_flows = grid.get("flows")
            if grid_flows:
                cfg["flows"] = copy.deepcopy(grid_flows)
            grid_features = grid.get("features")
            if grid_features:
                cfg["features"] = copy.deepcopy(grid_features)
            exp_input = exp.get("input")
            if exp_input:
                cfg.setdefault("input", {}).update(copy.deepcopy(exp_input))
            if explicit_name:
                cfg["output"]["experiment_name"] = explicit_name
            else:
                cfg["output"]["experiment_name"] = exp_name
            # Use provided preprocessed CSV if specified in grid
            if "input" in grid and "preprocessed_csv" in grid["input"]:
                cfg.setdefault("input", {})["preprocessed_csv"] = grid["input"]["preprocessed_csv"]

            # Write temp config and run
            tmp_cfg_path = Path(f".tmp_grid_cfg_{exp_name}_{idx}.yaml")
            tmp_cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
            try:
                run_experiment(tmp_cfg_path)
            except Exception as exc:
                print(f"[error] experiment failed: {cfg['output']['experiment_name']} -> {exc}", file=sys.stderr)
            finally:
                tmp_cfg_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a grid of clustering experiments.")
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("experiments/experiment_grid.yaml"),
        help="Path to the experiment grid YAML.",
    )
    args = parser.parse_args()
    main(args.grid)
