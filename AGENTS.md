# Working in this repository

This repository is the shared starting point for MaskedKD experiments. Keep it small and easy to clone.

- Read `README.md`, the relevant file in `datasets/`, and `GPU.md` before changing an experiment.
- Use the dataset's recorded split. Do not select methods or checkpoints using test results.
- Record the dataset name and manifest or metadata SHA-256, model initialization, teacher checkpoint, seed, patch budget, metric, and code commit with every reported result. Keep one row per seed and checkpoint.
- Do not commit source dataset images, segmentation masks, credentials, or raw training outputs. Only verified author-trained MaskedKD baseline weights belong in `checkpoints/`; register each file and its SHA-256 in `checkpoints/index.csv`.
- Check free GPU memory before training. Benchmark safe concurrency on the target machine; stop and resume from checkpoints when necessary. Do not interrupt another user's GPU job.
- Do not silently change a dataset split. Record the changed manifest or metadata SHA-256 and update the dataset documentation and affected results.
- Prefer small, documented changes. Run `python3 scripts/check_assets.py` on any dataset or checkpoint paths you use.
