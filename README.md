<p align="center">
  <img src="assets/kshs-aimlab-lockup-balanced.png" alt="강원과학고등학교 × AIM Lab" width="640">
</p>

<p align="center"><a href="README.md">English</a> · <a href="docs/ko/README.md">한국어</a></p>

<h1 align="center">KSHS × AIM Lab Benchmarks</h1>

<p align="center">Benchmarks for efficient vision distillation</p>

A benchmark for comparing the performance of MaskedKD and new methods.
[COCO single](datasets/coco_single/README.md) evaluates classification accuracy, while [Waterbirds](datasets/waterbirds/README.md) tests robustness to spurious correlations.

## Datasets

| Dataset | Task | Primary metric |
| --- | --- | --- |
| [COCO single](datasets/coco_single/README.md) | 10-class image classification | Macro accuracy |
| [Waterbirds](datasets/waterbirds/README.md) | Spurious correlation robustness | Worst-group accuracy |
| [ImageNet-1K](datasets/imagenet/README.md) | Pretraining and validation | Validation top-1/top-5 accuracy |

For data preparation and validation, see the [dataset guide](datasets/README.md).

## Getting Started

```bash
git clone https://github.com/jeehoo0507/kshs-aimlab-benchmarks.git
cd kshs-aimlab-benchmarks
bash scripts/fetch_maskedkd.sh
```

This fetches the official MaskedKD code. For Python dependencies, see the [official installation guide](https://github.com/effl-lab/MaskedKD#installation).

### Dataset Validation

```bash
python3 scripts/check_assets.py coco /path/to/coco_single
python3 scripts/check_assets.py waterbirds /path/to/waterbird_complete95_forest2water2
```

These commands check existing datasets; they do not download data.

## Checkpoints

The current MaskedKD checkpoints contain **DeiT-Tiny student** weights distilled from a DeiT-Small teacher. DeiT-Small student checkpoints will be added later.

| Student | Dataset (checkpoint) |
| --- | --- |
| DeiT-Tiny | [COCO single](checkpoints/coco/maskedkd_coco.pt) |
| DeiT-Tiny | [Waterbirds](checkpoints/waterbirds/maskedkd_waterbirds.pt) |
| DeiT-Small | — |
| DeiT-Small | — |

The [checkpoint guide](checkpoints/README.md) also provides a command to download official ImageNet-pretrained DeiT-Tiny, Small, and Base weights.

## Results

Measured MaskedKD results use a DeiT-Small teacher and DeiT-Tiny students (seeds 0, 1, 2). COCO's primary metric is macro accuracy; Waterbirds' is worst-group accuracy.

| Result | COCO single | Waterbirds | What it shows |
| --- | --- | --- | --- |
| Runs | [CSV](results/coco/runs.csv) | [CSV](results/waterbirds/runs.csv) | Validation-selected test scores by seed, training time, peak VRAM, and run provenance. |
| Summary | [CSV](results/coco/summary.csv) | [CSV](results/waterbirds/summary.csv) | Three-seed mean and standard deviation, plus isolated seed-0 training cost. |
| Epochs | [CSV](results/coco/epochs.csv) | [CSV](results/waterbirds/epochs.csv) | Training loss and validation accuracy across epochs. |
| Class / group | [Per-class CSV](results/coco/per_class.csv) | [Per-group CSV](results/waterbirds/per_group.csv) | Test accuracy by COCO class or Waterbirds bird/background group. |
| Validation masks | [CSV](results/coco/mask_validation.csv) | [CSV](results/waterbirds/mask_validation.csv) | Patch-selection changes, foreground coverage, and teacher/student prediction differences over training. |
| Test masks | [CSV](results/coco/mask_test.csv) | [CSV](results/waterbirds/mask_test.csv) | The selected checkpoint's foreground coverage and prediction disagreement among the student, full teacher, and masked teacher. |

See the [results guide](results/README.md) for metric definitions and comparison conditions.

## References

- [MaskedKD paper](https://arxiv.org/abs/2302.10494)
- [Official MaskedKD implementation](https://github.com/effl-lab/MaskedKD)
