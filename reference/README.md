# MaskedKD reference run

This runner creates the **DeiT-S → DeiT-Tiny** reference for COCO single and Waterbirds on one CUDA GPU. It implements the student-attention top-98 teacher mask used by MaskedKD; it is an adaptation to these two classification datasets, not a reproduction of the paper's ImageNet-1K DeiT-B → DeiT-S experiment. The [official MaskedKD source](https://github.com/effl-lab/MaskedKD) remains available through `scripts/fetch_maskedkd.sh` for inspection.

From a fresh clone of this branch on the GPU server:

```bash
git clone --branch codex/maskedkd-reference-runs https://github.com/jeehoo0507/kshs-aimlab-benchmarks.git
cd kshs-aimlab-benchmarks
bash scripts/run_reference.sh --dataset coco
bash scripts/run_reference.sh --dataset waterbirds
```

Run the last two commands separately; each resumes an interrupted run. The script creates `.venv/`, installs CUDA PyTorch 2.5.1, prepares missing data in the Git-ignored `data/` directory, validates it, caches official ImageNet DeiT-S and DeiT-Tiny weights, trains, evaluates, and exports CSVs. The first COCO setup may take substantial download time. To reuse data already on the server, pass `--data-root /absolute/path/to/coco_single` or `--data-root /absolute/path/to/waterbird_complete95_forest2water2`. The older `aim-lab-test-2` COCO subset is incompatible; the accepted subset has exactly one annotated object per image and includes `stop sign` rather than `cow`. To use an existing compatible Python environment, set `REFERENCE_PYTHON=/absolute/path/to/python`.

The fixed protocol in [`configs/reference.json`](../configs/reference.json) trains one ImageNet-initialized DeiT-S teacher for 30 epochs per dataset, then three ImageNet-initialized DeiT-Tiny students for 100 epochs (seeds 0, 1, 2). The teacher is frozen during KD. Teacher and student classifiers are replaced for 10 COCO or 2 Waterbirds classes; the student sees all 196 image patches while the teacher sees its selected 98. COCO selects checkpoints by validation macro accuracy and Waterbirds by validation worst-group accuracy; ties keep the earlier epoch. Test is evaluated only after selection.

The runner measures teacher VRAM before its first training job. Seed 0 runs alone for a clean student cost reference; the runner measures student peak VRAM with the actual model and batch size, then runs seeds 1 and 2 concurrently only if the measured capacity plus reserve fits the currently free GPU memory. Use `--jobs 1` for fully sequential student runs. No process already using the GPU is stopped. Training saves `resume.pt` every epoch in `outputs/reference/`; rerunning the command continues incomplete runs and reuses completed ones. View progress with `tail -f outputs/reference/coco/maskedkd/seed_0/train.log` or `.venv/bin/python -m reference.cli status --dataset coco`.

Finished runs populate the [`results/`](../results/README.md) CSVs. Student seed 0's validation-selected checkpoint is copied to `checkpoints/<dataset>/maskedkd_seed0.pt` for review; it is **not committed automatically**. The frozen teacher and all three full run directories remain under Git-ignored `outputs/reference/`; preserve the teacher checkpoint for future methods. The CSV records its SHA-256. Before another machine can make an exact comparison, publish that teacher artifact at a stable location or rerun the same teacher protocol and verify its hash.
