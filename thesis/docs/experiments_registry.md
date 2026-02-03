# Experiments Registry

This registry records standardized clustering experiments and their purpose.

Columns:
- id: experiment number (e.g., EXP001)
- question: research question / ablation axis
- preprocessed_id: link to preprocessed registry (e.g., preprocessed_1)
- distance: euclidean / dtw / frechet
- algorithm: kmeans / optics / hdbscan / etc.
- params: key parameter changes
- notes: optional remarks

| id | uid | question | preprocessed_id | distance | algorithm | params | notes |
|---|---|---|---|---|---|---|---|
| EXP001–EXP050 | see plan | See plan | preprocessed_1..7 | euclidean/dtw/frechet | mixed | see `exp01_50_plan.md` | Detailed matrix in plan |
