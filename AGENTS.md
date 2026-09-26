# Working in this repository

This repository is the shared starting point for MaskedKD experiments. Keep it small and easy to clone.

- Read `README.md`, the relevant file in `datasets/`, and `GPU.md` before changing an experiment.
- Use the dataset's recorded split. Do not select methods or checkpoints using test results.
- Record the dataset name and manifest or metadata SHA-256, model initialization, teacher checkpoint, seed, patch budget, metric, and code commit with every reported result. Keep one row per seed and checkpoint.
- Do not commit source dataset images, segmentation masks, credentials, or raw training outputs. Keep only the author-trained MaskedKD comparison checkpoint in each of `checkpoints/coco/` and `checkpoints/waterbirds/`.
- Check free GPU memory before training. Benchmark safe concurrency on the target machine; stop and resume from checkpoints when necessary. Do not interrupt another user's GPU job.
- Do not silently change a dataset split. Record the changed manifest or metadata SHA-256 and update the dataset documentation and affected results.
- Prefer small, documented changes. Run `python3 scripts/check_assets.py` on any dataset paths you use.
- The MaskedKD reference protocol lives in `configs/reference.json` and `reference/README.md`. Fill `results/` with its measured teacher and student runs before reporting a new method. Compare against `method=maskedkd` with the same dataset hash, frozen teacher hash, student initialization, 98-patch teacher budget, training epochs, and checkpoint-selection metric. Report accuracy, training cost, and the available learning-curve and mask-probe diagnostics; do not claim a cost gain from runs measured under different GPU concurrency.
