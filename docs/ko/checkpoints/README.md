# 체크포인트

[English](../../../checkpoints/README.md) · 한국어

저자가 직접 학습한 MaskedKD 비교 기준 체크포인트입니다.

```text
checkpoints/
├── coco/maskedkd_coco.pt
└── waterbirds/maskedkd_waterbirds.pt
```

두 파일에는 validation 기준으로 선택한 seed 0의 Student 가중치가 들어 있습니다. Seed 1과 2의 측정값은 [결과](../results/README.md)에 기록되어 있으며, 해당 가중치는 여기에 공개하지 않았습니다.

[측정 결과](../results/README.md)와 [기준 실험 절차](https://github.com/jeehoo0507/kshs-aimlab-benchmarks/tree/codex/maskedkd-reference-runs/reference)를 참고하세요.

## 공식 ImageNet 초기화 가중치

필요할 때 공식 **비증류(non-distilled)** DeiT 체크포인트를 다운로드하세요.

```bash
python3 scripts/fetch_deit_imagenet.py tiny small base
```

| 모델 | 현재 기준 실험에서의 역할 | 공식 체크포인트 |
| --- | --- | --- |
| DeiT-Tiny | Student | [ImageNet 가중치](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth) |
| DeiT-Small | Teacher | [ImageNet 가중치](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) |
| DeiT-Base | 아직 사용하지 않음 | [ImageNet 가중치](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) |

이 명령에는 PyTorch가 필요합니다. 다운로드한 파일의 SHA-256 접두사를 확인한 뒤 PyTorch hub 디렉터리(기본값 `~/.cache/torch/hub/checkpoints/`)에 캐시합니다. 외부에서 제공하는 원본 가중치는 이 저장소에 커밋하지 않습니다. 출처는 [공식 DeiT 모델 정의](https://github.com/facebookresearch/deit/blob/main/models.py)입니다.
