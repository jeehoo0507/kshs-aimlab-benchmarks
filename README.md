<p align="center">
  <img src="assets/kshs-aimlab-lockup-balanced.png" alt="KSHS × AIM Lab" width="640">
</p>

<h1 align="center">MaskedKD Benchmarks</h1>

<p align="center">KSHS × AIM Lab</p>

A benchmark for studying patch selection in MaskedKD on COCO single and Waterbirds. This repository documents the dataset protocols and provides a place for author-trained baseline checkpoints and evaluation results.

## Datasets

| Dataset | Task | Metric |
| --- | --- | --- |
| [COCO single](datasets/coco_single/README.md) | 10-class image classification | Macro accuracy |
| [Waterbirds](datasets/waterbirds/README.md) | Spurious correlation robustness | Worst-group accuracy |

Dataset images are not included. See the [data guide](datasets/README.md) for preparation and validation.

## Getting started

```bash
git clone https://github.com/jeehoo0507/kshs-aimlab-benchmarks.git
cd kshs-aimlab-benchmarks
bash scripts/fetch_maskedkd.sh
python3 scripts/check_assets.py coco /path/to/coco_single
python3 scripts/check_assets.py waterbirds /path/to/waterbird_complete95_forest2water2
```

The fetch script pins the [official MaskedKD implementation](baselines/maskedkd/README.md). The dataset commands validate files you have already prepared; they do not download data or train models.

## Checkpoints and results

Author-trained MaskedKD weights belong in [checkpoints](checkpoints/README.md); per-seed metrics and training cost belong in [results](results/README.md). Neither has been published yet. See the [GPU guide](GPU.md) before running experiments.

## References

- [MaskedKD paper](https://arxiv.org/abs/2302.10494)
- [Official implementation](https://github.com/effl-lab/MaskedKD)
