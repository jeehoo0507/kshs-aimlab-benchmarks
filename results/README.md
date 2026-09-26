# Results

[`coco.csv`](coco.csv) and [`waterbirds.csv`](waterbirds.csv) use the same columns. Add one row per seed, evaluation checkpoint, and metric; do not enter a multi-seed average as a run.

- `dataset` and `data_sha256` identify the dataset and its manifest (`COCO single`) or metadata (`Waterbirds`).
- `method`, `teacher_id`, `student_arch`, `student_init`, `keep_patches`, and `seed` identify the training setup.
- `eval_split`, `checkpoint`, `metric`, and `value` identify the evaluation.
- `train_hours`, `peak_vram_gib`, `code_commit`, and `run_ref` record cost and provenance.

The primary metrics are macro accuracy for COCO single and worst-group accuracy for Waterbirds. Compare runs only under matching data, teacher, initialization, patch budget, and checkpoint-selection rules.
