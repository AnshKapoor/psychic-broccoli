# Flight Trajectory ML Project

Thesis codebase for flight trajectory clustering and noise simulation.

## Commands

- Full pipeline (backbone clustering): `python cli.py -c config/backbone_full.yaml`
- Save preprocessed data only: `python scripts/save_preprocessed.py -c config/backbone_full.yaml`
- Run extended experiment (cluster + metrics): `python experiments/runner.py -c config/backbone_full.yaml`
- You can override the preprocessed CSV for the experiment runner with `--preprocessed path/to/preprocessed_xxx.csv`, or it will pick `input.preprocessed_csv` from the config, or the latest `preprocessed_*.csv` under `output/run/csv`.
- Run experiment grid sweeps (OPTICS/DBSCAN/HDBSCAN/KMeans/Agglomerative): `python scripts/run_experiment_grid.py`
