"""Run a grid of clustering experiments defined in experiments/experiment_grid.yaml."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import yaml

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
    base_name = grid.get("output", {}).get("base_experiment_name", "grid")
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

        for idx, params in enumerate(param_space, start=1):
            cfg = copy.deepcopy(base_cfg)
            cfg["clustering"]["method"] = method
            cfg["clustering"]["distance_metric"] = distance_metric
            cfg["clustering"][method] = params
            cfg["output"]["experiment_name"] = f"{base_name}_{exp_name}_{method}_run{idx}"
            # Use provided preprocessed CSV if specified in grid
            if "input" in grid and "preprocessed_csv" in grid["input"]:
                cfg.setdefault("input", {})["preprocessed_csv"] = grid["input"]["preprocessed_csv"]

            # Write temp config and run
            tmp_cfg_path = Path(f".tmp_grid_cfg_{exp_name}_{idx}.yaml")
            tmp_cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
            run_experiment(tmp_cfg_path)
            tmp_cfg_path.unlink()


if __name__ == "__main__":
    main(Path("experiments/experiment_grid.yaml"))
