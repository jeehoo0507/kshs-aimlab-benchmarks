<p align="center">
  <img src="assets/kshs-aimlab-lockup-balanced.png" alt="강원과학고등학교 × AIM Lab" width="640">
</p>

<p align="center"><a href="README.md">English</a> · <a href="README.ko.md">한국어</a></p>

<h1 align="center">KSHS × AIM Lab Benchmarks</h1>

<p align="center">Benchmarks for efficient vision distillation</p>

A benchmark for comparing the performance of MaskedKD and new methods.
[COCO single](datasets/coco_single/README.md) evaluates classification accuracy, while [Waterbirds](datasets/waterbirds/README.md) tests robustness to spurious correlations.

## Datasets

| Dataset | Task | Primary metric |
| --- | --- | --- |
| [COCO single](datasets/coco_single/README.md) | 10-class image classification | Macro accuracy |
| [Waterbirds](datasets/waterbirds/README.md) | Spurious correlation robustness | Worst-group accuracy |

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

Author-trained MaskedKD reference checkpoints are available for [COCO single](checkpoints/coco/maskedkd.pt) and [Waterbirds](checkpoints/waterbirds/maskedkd.pt).

## Results

- [COCO single results](results/coco.csv): per-seed macro accuracy and training cost.
- [Waterbirds results](results/waterbirds.csv): per-seed worst-group accuracy and training cost.

The [results guide](results/README.md) links the three-seed summaries, per-class/group accuracy, learning curves, and mask diagnostics.

## References

- [MaskedKD paper](https://arxiv.org/abs/2302.10494)
- [Official MaskedKD implementation](https://github.com/effl-lab/MaskedKD)
