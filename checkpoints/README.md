# Checkpoints

This directory is for author-trained MaskedKD baseline weights. No weights have been registered yet.

Add each file under `checkpoints/maskedkd/<dataset>/` and register it in [`index.csv`](index.csv). Each row records its ID, dataset, architecture, teacher/student role, seed, source run, relative path (from `checkpoints/`), and SHA-256.

Verify the published files from the repository root:

```bash
python3 scripts/check_assets.py checkpoints --full
```

Use `--root /path/to/checkpoints` to verify files outside this checkout.
