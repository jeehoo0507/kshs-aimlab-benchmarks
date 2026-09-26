<p align="center">
  <img src="assets/kshs-aimlab-lockup.png" alt="강원과학고등학교 × AIM Lab" width="640">
</p>

<h1 align="center">KSHS × AIM Lab</h1>

<p align="center"><strong>Benchmarks for efficient vision distillation</strong></p>

<p align="center">
  <a href="datasets/README.md">Datasets</a> ·
  <a href="baselines/maskedkd/README.md">MaskedKD</a> ·
  <a href="checkpoints/README.md">Checkpoints</a> ·
  <a href="results/README.md">Results</a> ·
  <a href="GPU.md">GPU guide</a>
</p>

---

MaskedKD를 출발점으로 Vision Transformer 지식 증류의 **정확도와 계산 비용**을 함께 비교합니다. 이 저장소는 팀원이 같은 데이터 분할과 평가 기준으로 실험할 수 있도록 데이터 정의, 기준 코드, 체크포인트 목록, 결과 형식을 모읍니다.

## Benchmarks

| Dataset | 평가 목적 | 주요 지표 |
| --- | --- | --- |
| [COCO single v1](datasets/coco_single_v1/README.md) | 10-class 분류와 patch 선택 | Macro accuracy |
| [Waterbirds v1](datasets/waterbirds_v1/README.md) | 새 종류·배경 간 허위 상관관계 | Worst-group accuracy |

이미지와 모델 가중치는 Git에 포함하지 않습니다. 각 데이터셋 문서에 구성과 필요한 파일을 적고, 체크포인트는 해시와 출처로 관리합니다.

## Getting started

데이터가 준비되면 Python 표준 라이브러리만으로 구성을 확인할 수 있습니다.

```bash
python3 scripts/check_assets.py coco /absolute/path/to/coco_single
python3 scripts/check_assets.py waterbirds /absolute/path/to/waterbird_complete95_forest2water2
python3 scripts/check_assets.py checkpoints
```

FG/BG 분석용 Waterbirds segmentation을 확인할 때는 `--seg-root`를 추가합니다. 공식 기준 코드는 `bash scripts/fetch_maskedkd.sh`로 가져옵니다. 새 실험을 시작할 때는 [팀 작업 규칙](AGENTS.md)을 확인하세요.

## References

- [MaskedKD paper — The Role of Masking for Efficient Supervised Knowledge Distillation of Vision Transformers](https://arxiv.org/abs/2302.10494)
- [Official MaskedKD implementation](https://github.com/effl-lab/MaskedKD)
