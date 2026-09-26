<p align="center">
  <img src="assets/kshs-aimlab-lockup-balanced.png" alt="강원과학고등학교 × AIM Lab" width="640">
</p>

<h1 align="center">KSHS × AIM Lab Benchmarks</h1>

<p align="center">Benchmarks for efficient vision distillation</p>

COCO single과 Waterbirds에서 MaskedKD의 패치 선택 방식을 비교하기 위한 저장소입니다. 데이터 구성 기준과 검증 도구를 제공하며, 직접 학습한 비교용 MaskedKD 체크포인트와 평가 결과를 추가할 예정입니다.

## 데이터셋

| 데이터셋 | 평가 과제 | 주요 지표 |
| --- | --- | --- |
| [COCO single](datasets/coco_single/README.md) | 10개 클래스 이미지 분류 | 매크로 정확도 |
| [Waterbirds](datasets/waterbirds/README.md) | 배경과 라벨의 허위 상관관계 평가 | 최저 그룹 정확도 |

원본 이미지는 포함하지 않습니다. 준비 및 검증 방법은 [데이터셋 안내](datasets/README.md)를 참고하세요.

## 시작하기

```bash
git clone https://github.com/jeehoo0507/kshs-aimlab-benchmarks.git
cd kshs-aimlab-benchmarks
python3 scripts/check_assets.py coco /path/to/coco_single
python3 scripts/check_assets.py waterbirds /path/to/waterbird_complete95_forest2water2
```

위 명령은 준비된 데이터 파일을 확인합니다. 데이터 다운로드나 모델 학습은 수행하지 않습니다. [MaskedKD 공식 코드](baselines/maskedkd/README.md)가 필요하면 `bash scripts/fetch_maskedkd.sh`를 실행하세요.

## 체크포인트와 결과

비교용 MaskedKD 가중치는 [체크포인트](checkpoints/README.md)에, 무작위 시드별 정확도와 학습 비용은 [결과](results/README.md)에 기록합니다. 실제 체크포인트와 결과는 아직 등록되지 않았습니다. 실험 전에는 [GPU 가이드](GPU.md)를 확인하세요.

## 참고 자료

- [MaskedKD 논문](https://arxiv.org/abs/2302.10494)
- [MaskedKD 공식 구현](https://github.com/effl-lab/MaskedKD)
