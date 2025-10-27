prepare:
	python scripts/prepare_dataset.py --config config/paths.yaml

extract:
	python scripts/extract_trajectories.py --config config/paths.yaml

cluster:
	python scripts/run_clustering.py --config config/params.yaml
