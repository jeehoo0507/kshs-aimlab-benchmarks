<p align="center">
  <img src="assets/school-logo.png" alt="logo" width="160">
</p>

<h1 align="center">AIM Lab</h1>

<p align="center">Vision Transformer 지식 증류를 더 효율적이고 신뢰할 수 있게</p>

## 소개

이 저장소는 MaskedKD를 바탕으로 한 연구를 팀에서 공유하는 작업 공간입니다. Student attention으로 teacher가 볼 패치를 고르는 과정과 그에 따른 정확도·연산 비용의 변화를 살펴봅니다. COCO와 Waterbirds에서 같은 기준으로 실험을 비교할 수 있도록 자료를 정리합니다.

## 이 저장소에서 공유하는 것

- **[데이터셋](datasets/README.md)** — 이미지 목록, 분할 기준, 전처리 방법
- **[기준 모델](baselines/maskedkd/README.md)과 [체크포인트](checkpoints/README.md)** — 공식 MaskedKD 코드와 가중치의 출처
- **[실험 결과](results/README.md)** — 조건과 seed별 정확도, 실행 시간, GPU 사용량
- **작업 가이드** — [팀 공통 규칙](AGENTS.md)과 [GPU 사용법](GPU.md)

대용량 이미지와 모델 가중치는 별도 저장 공간에 두고, 이 저장소에는 이를 확인하고 다시 사용할 수 있는 정보와 코드를 모읍니다.

## 시작하기

데이터를 준비했다면 파일 위치를 지정해 구성을 확인합니다. 점검 도구는 Python 표준 라이브러리만 사용하며 GPU가 필요하지 않습니다.

```bash
python3 scripts/check_assets.py coco /absolute/path/to/coco_single
python3 scripts/check_assets.py waterbirds /absolute/path/to/waterbird_complete95_forest2water2
python3 scripts/check_assets.py checkpoints
```

FG/BG 분석에 사용할 Waterbirds segmentation은 `--seg-root`를 추가해 확인합니다. 공식 MaskedKD 코드는 `bash scripts/fetch_maskedkd.sh`로 고정된 revision을 받아올 수 있습니다. 학습 명령은 새 실험을 구현할 때 해당 실험 문서에 기록합니다.

## 참고 자료

- [MaskedKD 논문 — The Role of Masking for Efficient Supervised Knowledge Distillation of Vision Transformers](https://arxiv.org/abs/2302.10494)
- [MaskedKD 공식 구현 — effl-lab/MaskedKD](https://github.com/effl-lab/MaskedKD)
