# GPU guide

Check available memory and other processes before starting a run:

```bash
nvidia-smi
watch -n 2 nvidia-smi
```

On an RTX A5000, the 24 GiB total is not necessarily available. Benchmark one job, then compare throughput with two and four concurrent jobs. Leave 2–3 GiB for CUDA initialization and other processes; use the highest concurrency that finishes without OOM.

Resume from the latest saved epoch with the same data, teacher checkpoint, and configuration. Record training time and peak VRAM alongside accuracy. `scripts/check_assets.py` only validates files and does not use the GPU.

The [MaskedKD reference runner](reference/README.md) trains seed 0 alone, benchmarks its actual student VRAM, then runs seeds 1 and 2 only within the measured free-memory budget. Use `--jobs 1` to force sequential student training. Logs and per-epoch resume checkpoints are in `outputs/reference/`.
