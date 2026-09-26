# Working in this repository

This repository is the shared starting point for MaskedKD experiments. Keep it small and easy to clone.

- Read `README.md`, the relevant file in `datasets/`, and `GPU.md` before changing an experiment.
- Use the dataset's recorded split. Do not select methods or checkpoints using test results.
- Record the data version, model initialization, teacher checkpoint, seed, patch budget, metric, and code commit with every reported result. Keep one row per seed and checkpoint.
- Do not commit images, segmentation masks, model weights, credentials, or raw training outputs. Register weights in `checkpoints/index.csv` and keep large files in team storage.
- Check free GPU memory before training. Benchmark safe concurrency on the target machine; stop and resume from checkpoints when necessary. Do not interrupt another user's GPU job.
- When a dataset definition changes, create a new versioned dataset directory instead of silently changing an existing split.
- Prefer small, documented changes. Run `python3 scripts/check_assets.py` on any dataset or checkpoint paths you use.

