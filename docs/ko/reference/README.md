# MaskedKD 기준 실험

[English](../../../reference/README.md) · 한국어

이 실행기는 단일 CUDA GPU에서 COCO single과 Waterbirds용 **DeiT-S → DeiT-Tiny** 기준 실험을 만듭니다. Student의 attention 상위 98패치만 Teacher가 보는 MaskedKD 마스킹을 구현했습니다. 두 분류 데이터셋에 맞게 조정한 실험이며, 논문의 ImageNet-1K DeiT-B → DeiT-S 실험을 그대로 재현한 것은 아닙니다. [MaskedKD 공식 코드](https://github.com/effl-lab/MaskedKD)는 `scripts/fetch_maskedkd.sh`로 받아 확인할 수 있습니다.

GPU 서버에서 이 브랜치를 새로 클론했다면 다음 명령을 실행하세요.

```bash
git clone --branch codex/maskedkd-reference-runs https://github.com/jeehoo0507/kshs-aimlab-benchmarks.git
cd kshs-aimlab-benchmarks
bash scripts/run_reference.sh --dataset coco
bash scripts/run_reference.sh --dataset waterbirds
```

마지막 두 명령은 각각 실행하세요. 중단한 실험은 같은 명령으로 재개할 수 있습니다. 스크립트는 `.venv/` 생성과 CUDA PyTorch 2.5.1 설치, Git에서 제외된 `data/`의 누락 데이터 준비·검증, 공식 ImageNet DeiT-S·DeiT-Tiny 가중치 캐시, 학습·평가·CSV 내보내기를 수행합니다. COCO를 처음 준비할 때는 다운로드에 시간이 걸릴 수 있습니다. 서버에 이미 데이터가 있으면 `--data-root /absolute/path/to/coco_single` 또는 `--data-root /absolute/path/to/waterbird_complete95_forest2water2`를 지정하세요. 예전 `aim-lab-test-2` COCO 부분집합은 호환되지 않습니다. 이 실험에서 허용하는 부분집합은 이미지당 주석 객체가 정확히 하나이고 `cow` 대신 `stop sign`을 포함합니다. 기존 Python 환경을 쓰려면 `REFERENCE_PYTHON=/absolute/path/to/python`을 설정하세요.

Waterbirds 전경 분석에는 [공식 CUB segmentation 마스크](https://data.caltech.edu/records/w9d68-gec53)도 다운로드하여 모든 합성 이미지에 대응하는 마스크가 있는지 확인합니다. 서버에 마스크가 있다면 `--seg-root /absolute/path/to/CUB_200_2011/segmentations`를 추가하세요. Validation 진단은 설정된 epoch마다 고정 validation 전체를 사용합니다. 마지막 test 진단은 validation으로 선택한 최적 체크포인트를 사용하며 이미지별 패치 마스크와 예측을 `outputs/reference/<dataset>/maskedkd/seed_<seed>/test_mask.json`에 남깁니다.

고정 절차인 [`configs/reference.json`](../../../configs/reference.json)에 따라 데이터셋마다 ImageNet으로 초기화한 DeiT-S Teacher를 30 epoch 학습하고, ImageNet으로 초기화한 DeiT-Tiny Student 세 개(seed 0, 1, 2)를 각각 100 epoch 학습합니다. KD 중 Teacher는 고정합니다. 분류 head는 COCO 10개 또는 Waterbirds 2개 클래스에 맞게 교체합니다. Student는 이미지 패치 196개를 모두 보고 Teacher는 선택된 98개만 봅니다. COCO는 validation macro accuracy, Waterbirds는 validation worst-group accuracy로 체크포인트를 고릅니다. 동점이면 더 이른 epoch를 유지하고, test는 선택 후에만 평가합니다.

첫 학습 작업 전에 Teacher VRAM을 측정합니다. Student seed 0은 비교 가능한 비용 기준을 위해 단독으로 실행합니다. 이후 실제 모델과 batch size로 Student peak VRAM을 측정하고, 현재 빈 VRAM에서 전역 여유 2.5 GiB를 남길 수 있을 때만 seed 1과 2를 병렬 실행합니다. 최대 2개라는 제한은 남은 seed 수 때문이며 GPU의 고정 제한이 아닙니다. Student를 모두 순차 실행하려면 `--jobs 1`을 사용하세요. 이미 GPU를 쓰는 다른 프로세스는 중단하지 않습니다. 매 epoch의 `resume.pt`로 중단된 실행을 이어가고 완료된 실행은 재사용합니다. 진행 상태는 `tail -f outputs/reference/coco/maskedkd/seed_0/train.log` 또는 `.venv/bin/python -m reference.cli status --dataset coco`로 볼 수 있습니다.

완료된 실행은 [`results/`](../results/README.md)의 CSV에 기록됩니다. Validation으로 선택한 Student seed 0 체크포인트는 검토용으로 `checkpoints/<dataset>/maskedkd_<dataset>.pt`에 복사하지만 **자동 커밋하지 않습니다**. 고정 Teacher와 Student 세 seed의 전체 실행 디렉터리는 Git에서 제외된 `outputs/reference/`에 남습니다. 이후 방법을 비교할 수 있도록 Teacher 체크포인트를 보존하세요. 해당 SHA-256은 CSV에 기록됩니다. 다른 컴퓨터에서 정확히 같은 비교를 하려면 Teacher 가중치를 안정적인 위치에 공개하거나 같은 절차로 다시 학습하고 해시를 확인해야 합니다.
