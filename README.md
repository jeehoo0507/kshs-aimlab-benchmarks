<p align="center">
  <img src="assets/kshs-aimlab-lockup-balanced.png" alt="강원과학고등학교 × AIM Lab" width="640">
</p>

<h1 align="center">KSHS × AIM Lab Benchmarks</h1>

<p align="center">Vision Transformer 지식 증류 실험을 위한 공통 데이터·기준·결과 기록</p>

## 소개

이 저장소는 [MaskedKD](https://arxiv.org/abs/2302.10494)를 기준으로 patch 선택 방식의 정확도와 계산 비용을 비교하기 위한 공통 출발점입니다. 팀원이 같은 데이터 분할, teacher checkpoint, 평가 지표를 사용하도록 데이터 정의와 검증 도구를 모았습니다.

현재 제공하는 것은 **데이터셋 명세, 자산 검증 도구, 기준 코드의 고정 버전, 체크포인트·결과 기록 양식**입니다. 원본 데이터, 모델 가중치, 학습 결과와 학습 실행 코드는 포함하지 않습니다.

## 데이터셋

| 데이터셋 | 실험 목적 | 주요 평가 지표 |
| --- | --- | --- |
| [COCO single](datasets/coco_single/README.md) | 단일 카테고리 이미지의 10-class 분류 및 patch 선택 분석 | Macro accuracy |
| [Waterbirds](datasets/waterbirds/README.md) | 새 종류와 배경 사이의 허위 상관관계 평가 | Worst-group accuracy |

각 문서에 데이터 구성, 필요한 파일, 검증 명령을 적었습니다. 원본 이미지는 Git에 넣지 않고 로컬 `data/` 또는 팀 공유 저장 공간에 둡니다.

## 저장소 구성

| 경로 | 내용 |
| --- | --- |
| [`datasets/`](datasets/README.md) | COCO single·Waterbirds 데이터 정의 |
| [`baselines/maskedkd/`](baselines/maskedkd/README.md) | MaskedKD 공식 코드의 기준 commit과 가져오는 방법 |
| [`checkpoints/`](checkpoints/README.md) | teacher·student 가중치 출처와 해시를 기록할 목록 |
| [`results/`](results/README.md) | seed별 정확도·학습 시간·VRAM 기록 양식 |
| [`scripts/check_assets.py`](scripts/check_assets.py) | 데이터 파일과 체크포인트 확인 도구 |
| [`GPU.md`](GPU.md) | GPU 사용 및 동시 실행 가이드 |

## 시작하기

저장소를 클론한 뒤, [데이터셋 문서](datasets/README.md)에 맞춰 준비한 파일을 확인합니다. 아래 경로는 자신의 로컬 경로로 바꿔 실행하세요. 검증 도구는 Python 표준 라이브러리만 사용하며 GPU가 필요하지 않습니다.

```bash
git clone https://github.com/jeehoo0507/kshs-aimlab-benchmarks.git
cd kshs-aimlab-benchmarks

python3 scripts/check_assets.py coco /absolute/path/to/coco_single
python3 scripts/check_assets.py waterbirds /absolute/path/to/waterbird_complete95_forest2water2
python3 scripts/check_assets.py checkpoints
```

기준 구현이 필요하면 `bash scripts/fetch_maskedkd.sh`로 [MaskedKD 공식 코드](baselines/maskedkd/README.md)를 고정된 commit에 가져옵니다. 실험 결과를 추가할 때는 [결과 형식](results/README.md)과 [팀 작업 규칙](AGENTS.md)을 확인하세요.

## 참고 문헌

- [MaskedKD 논문](https://arxiv.org/abs/2302.10494)
- [MaskedKD 공식 구현](https://github.com/effl-lab/MaskedKD)
