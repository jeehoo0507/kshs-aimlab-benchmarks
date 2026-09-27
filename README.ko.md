<p align="center">
  <img src="assets/kshs-aimlab-lockup-balanced.png" alt="강원과학고등학교 × AIM Lab" width="640">
</p>

<p align="center"><a href="README.md">English</a> · <a href="README.ko.md">한국어</a></p>

<h1 align="center">KSHS × AIM Lab Benchmarks</h1>

<p align="center">효율적인 비전 지식 증류를 위한 벤치마크</p>

MaskedKD와 새로운 방법의 성능을 비교하기 위한 벤치마크입니다.
[COCO single](datasets/coco_single/README.md)에서는 분류 정확도를, [Waterbirds](datasets/waterbirds/README.md)에서는 허위 상관관계에 대한 강건성을 평가합니다.

## 데이터셋

| 데이터셋 | 과제 | 주요 지표 |
| --- | --- | --- |
| [COCO single](datasets/coco_single/README.md) | 10개 클래스 이미지 분류 | 클래스별 정확도의 평균 (macro accuracy) |
| [Waterbirds](datasets/waterbirds/README.md) | 허위 상관관계에 대한 강건성 | 그룹별 정확도의 최솟값 (worst-group accuracy) |
| [ImageNet-1K](datasets/imagenet/README.md) | 사전학습 및 검증 | validation top-1/top-5 정확도 |

데이터 준비와 검증 방법은 [데이터셋 안내](datasets/README.md)를 참고하세요.

## 시작하기

```bash
git clone https://github.com/jeehoo0507/kshs-aimlab-benchmarks.git
cd kshs-aimlab-benchmarks
bash scripts/fetch_maskedkd.sh
```

MaskedKD 공식 코드를 가져옵니다. Python 의존성은 [공식 설치 안내](https://github.com/effl-lab/MaskedKD#installation)를 참고하세요.

### 데이터셋 확인

```bash
python3 scripts/check_assets.py coco /path/to/coco_single
python3 scripts/check_assets.py waterbirds /path/to/waterbird_complete95_forest2water2
```

이 명령은 이미 준비된 데이터셋을 확인하며, 다운로드는 수행하지 않습니다.

## 체크포인트

현재 MaskedKD 체크포인트는 DeiT-Small Teacher로 증류한 **DeiT-Tiny Student** 가중치입니다. DeiT-Small Student 체크포인트는 추후 추가할 예정입니다.

| Student | 데이터셋 (체크포인트) |
| --- | --- |
| DeiT-Tiny | [COCO single](checkpoints/coco/maskedkd_coco.pt) |
| DeiT-Tiny | [Waterbirds](checkpoints/waterbirds/maskedkd_waterbirds.pt) |
| DeiT-Small | — |
| DeiT-Small | — |

[체크포인트 안내](checkpoints/README.md)에는 공식 ImageNet 사전학습 DeiT-Tiny·Small·Base 가중치 다운로드 명령도 있습니다.

## 결과

DeiT-Small Teacher와 DeiT-Tiny Student(seed 0, 1, 2)로 측정한 MaskedKD 결과입니다. 주요 지표는 COCO의 macro accuracy와 Waterbirds의 worst-group accuracy입니다.

| 항목 | COCO single | Waterbirds | 내용 |
| --- | --- | --- | --- |
| 실행별 결과 | [CSV](results/coco/runs.csv) | [CSV](results/waterbirds/runs.csv) | seed별 validation 선택 checkpoint의 test 점수, 학습 시간, 최대 VRAM, 실행 정보. |
| 요약 | [CSV](results/coco/summary.csv) | [CSV](results/waterbirds/summary.csv) | 3개 seed의 평균·표준편차와 단독 실행한 seed 0의 학습 비용. |
| Epoch 기록 | [CSV](results/coco/epochs.csv) | [CSV](results/waterbirds/epochs.csv) | epoch별 학습 loss와 validation 정확도. |
| 클래스·그룹별 | [클래스 CSV](results/coco/per_class.csv) | [그룹 CSV](results/waterbirds/per_group.csv) | COCO 클래스별 또는 Waterbirds 새 종류·배경 그룹별 test 정확도. |
| Validation 마스크 | [CSV](results/coco/mask_validation.csv) | [CSV](results/waterbirds/mask_validation.csv) | 학습 중 패치 선택 변화, 전경 포함률, Teacher·Student 예측 차이. |
| Test 마스크 | [CSV](results/coco/mask_test.csv) | [CSV](results/waterbirds/mask_test.csv) | 선택된 checkpoint의 전경 포함률과 Student·전체 Teacher·마스킹 Teacher 간 예측 불일치. |

지표 정의와 비교 조건은 [결과 안내](results/README.md)를 참고하세요.

## 참고 자료

- [MaskedKD 논문](https://arxiv.org/abs/2302.10494)
- [MaskedKD 공식 구현](https://github.com/effl-lab/MaskedKD)
