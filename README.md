<p align="center">
  <img src="assets/kshs-aimlab-lockup-balanced.png" alt="강원과학고등학교 × AIM Lab" width="640">
</p>

<h1 align="center">KSHS × AIM Lab — MaskedKD Benchmarks</h1>

<p align="center">Dataset protocols · Baseline checkpoints · Evaluation results</p>

## Overview

COCO single과 Waterbirds에서 [MaskedKD](https://arxiv.org/abs/2302.10494)를 기준으로 patch 선택 방식의 정확도와 계산 비용을 비교합니다. 이 저장소에는 데이터 구성과 검증 방법, 저자가 직접 학습한 MaskedKD baseline checkpoint, 평가 결과를 정리합니다.

현재는 데이터 명세와 검증 도구가 준비되어 있습니다. Checkpoint 파일과 측정된 결과는 검증 후 추가합니다.

## Datasets

| Dataset | Task | Primary metric |
| --- | --- | --- |
| [COCO single](datasets/coco_single/README.md) | 10-class image classification | Macro accuracy |
| [Waterbirds](datasets/waterbirds/README.md) | Spurious correlation robustness | Worst-group accuracy |

원본 이미지와 segmentation mask는 저장소에 포함하지 않습니다. 데이터 선정 기준, split, 필요한 파일은 각 데이터셋 문서에 있습니다.

## Checkpoints

[`checkpoints/`](checkpoints/README.md)는 저자가 직접 학습한 MaskedKD baseline 가중치를 게시하는 위치입니다. Checkpoint마다 dataset, model, seed, 원본 실험 실행 기록과 SHA-256을 `index.csv`에 기록합니다. 현재 등록된 checkpoint는 없습니다.

## Quick start

저장소를 클론하고 준비한 데이터 경로를 검증합니다. 검증 스크립트는 Python 표준 라이브러리만 사용하며 GPU가 필요하지 않습니다.

```bash
git clone https://github.com/jeehoo0507/kshs-aimlab-benchmarks.git
cd kshs-aimlab-benchmarks

python3 scripts/check_assets.py coco /absolute/path/to/coco_single
python3 scripts/check_assets.py waterbirds /absolute/path/to/waterbird_complete95_forest2water2
python3 scripts/check_assets.py checkpoints --full
```

MaskedKD 공식 구현은 `bash scripts/fetch_maskedkd.sh`로 고정된 commit에 받을 수 있습니다. 데이터 검증 명령은 데이터 생성이나 모델 학습을 수행하지 않습니다.

## Repository guide

| Path | Description |
| --- | --- |
| [`datasets/`](datasets/README.md) | Dataset protocols and validation |
| [`baselines/maskedkd/`](baselines/maskedkd/README.md) | Pinned official MaskedKD source |
| [`checkpoints/`](checkpoints/README.md) | Author-trained MaskedKD baseline weights |
| [`results/`](results/README.md) | Per-seed metrics and training cost |
| [`GPU.md`](GPU.md) | GPU usage guide |

새 실험을 시작할 때는 [AGENTS.md](AGENTS.md)를 확인합니다.

## References

- [MaskedKD paper](https://arxiv.org/abs/2302.10494)
- [Official MaskedKD implementation](https://github.com/effl-lab/MaskedKD)
