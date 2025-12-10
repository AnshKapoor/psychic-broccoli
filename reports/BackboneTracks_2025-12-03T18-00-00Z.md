# Backbone Tracks Update — 2025-12-03 18:00 UTC

## Summary of new work
- Enabled UTM-based geometry (EPSG:32632) for smoothing, resampling, clustering, and backbone percentile computation.
- Added YAML config section `coordinates.use_utm` / `coordinates.utm_crs` to toggle projection.
- Introduced UTM conversion helper (`add_utm_coordinates`) and integrated it early in the pipeline.
- Updated preprocessing to smooth/resample `x_utm`, `y_utm` (with optional lat/lon retention).
- Updated clustering to build feature vectors from UTM coordinates.
- Updated backbone computation to emit UTM percentiles (and keep lat/lon where available).
- Updated plots to render in UTM space when enabled.
- Added logging config to write to `logs/backbone.log` with custom filename/dir.
- Documented control flow and logged changes with timestamps.

## Control flow (current pipeline)
1) Load YAML config; configure logging.
2) Load CSVs (with optional row cap in test mode); ensure required columns.
3) If `coordinates.use_utm` is true, convert lon/lat → `x_utm`, `y_utm` (EPSG:32632).
4) Optional flow filter (A/D + Runway) → segment flights (time gap, direction change, distance jump).
5) Optional test caps: limit flights per flow; sample flows-only if configured.
6) Preprocess: smooth (`x_utm`, `y_utm`, alt, etc.), resample to N points.
7) Per-flow clustering (OPTICS/KMeans) using UTM feature vectors.
8) Compute backbone percentiles per cluster/step (UTM + optional lat/lon; altitude; distance).
9) Save preprocessed, clustered, and backbone CSVs; optional plots (UTM if enabled).

## Files touched today
- `config/backbone.yaml`: added `coordinates` section; logging config; retained test-mode knobs.
- `backbone_tracks/io.py`: UTM converter helper.
- `backbone_tracks/preprocessing.py`: UTM-aware smoothing/resampling.
- `backbone_tracks/clustering.py`: UTM feature vectors.
- `backbone_tracks/backbone.py`: UTM percentiles in outputs.
- `backbone_tracks/plots.py`: UTM plotting option.
- `cli.py`: wired UTM conversion, passed flags through pipeline, logging config.
- `reports/BackboneTracks_1.md`: update log/timestamp.

## Notes
- Default CRS for UTM: EPSG:32632 (suitable for Hannover region).
- Test mode remains: row cap, flights-per-flow cap, flow sampling, cluster cap.
- Outputs remain in `output/` (preprocessed, clustered, backbone CSVs) and `logs/` for run logs.
