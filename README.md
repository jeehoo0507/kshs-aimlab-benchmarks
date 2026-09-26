<p align="center">
  <img src="assets/kshs-aimlab-lockup-balanced.png" alt="강원과학고등학교 × AIM Lab" width="640">
</p>

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

## 체크포인트와 결과

비교용 MaskedKD 가중치는 [체크포인트](checkpoints/README.md)에, 무작위 시드별 정확도와 학습 비용은 [결과](results/README.md)에 기록합니다. 실제 체크포인트와 결과는 아직 등록되지 않았습니다. 실험 전에는 [GPU 가이드](GPU.md)를 확인하세요.

## 참고 자료

- [MaskedKD 논문](https://arxiv.org/abs/2302.10494)
- [MaskedKD 공식 구현](https://github.com/effl-lab/MaskedKD)
