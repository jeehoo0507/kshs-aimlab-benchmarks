# GPU guide

Check available memory and other processes before starting a run:

```bash
nvidia-smi
watch -n 2 nvidia-smi
```

On an RTX A5000, the 24 GiB total is not necessarily available. Measure the peak VRAM of **one representative job**, then calculate how many pending jobs fit in the currently free memory:

```bash
python3 scripts/gpu_capacity.py --peak-gib 2.26 --pending 12
```

Replace `2.26` with the measured one-job peak and `12` with the number of independent jobs waiting. The command reads current free VRAM from `nvidia-smi` and returns a memory-limited maximum; it does not launch jobs. It leaves **2.5 GiB free globally** and budgets an additional 25% plus 0.5 GiB per job for allocator and process variation. There is no fixed 1/2/4-job ceiling. Launch jobs one at a time, recheck free VRAM after each allocation, and stop adding jobs before the reserve is crossed. Compare completed work per hour across feasible counts if throughput matters; the largest count is not always the fastest.

Resume from the latest saved epoch with the same data, teacher checkpoint, and configuration. Record training time and peak VRAM alongside accuracy. `scripts/check_assets.py` only validates files and does not use the GPU.

The [MaskedKD reference runner](https://github.com/jeehoo0507/kshs-aimlab-benchmarks/tree/codex/maskedkd-reference-runs/reference) measures the actual student VRAM and uses the same calculation with `--jobs auto`. It trains seed 0 alone for a comparable cost measurement, so only seeds 1 and 2 remain to run concurrently; **two is a limit of this experiment grid, not a GPU limit**. Use `--jobs 1` to force sequential student training. Logs and per-epoch resume checkpoints are in `outputs/reference/`.
