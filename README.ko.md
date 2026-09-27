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

직접 학습한 비교용 MaskedKD 체크포인트는 [COCO single](checkpoints/coco/maskedkd_coco.pt)과 [Waterbirds](checkpoints/waterbirds/maskedkd_waterbirds.pt)에 있습니다.

## 결과

- [COCO single 결과](results/coco/runs.csv): seed별 macro accuracy와 학습 비용.
- [Waterbirds 결과](results/waterbirds/runs.csv): seed별 worst-group accuracy와 학습 비용.

[결과 안내](results/README.md)에서 3개 seed 요약, 클래스·그룹별 정확도, 학습 곡선과 마스킹 진단을 볼 수 있습니다.

## 참고 자료

- [MaskedKD 논문](https://arxiv.org/abs/2302.10494)
- [MaskedKD 공식 구현](https://github.com/effl-lab/MaskedKD)
