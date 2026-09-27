# Checkpoints

MaskedKD checkpoints trained by the author for baseline comparison.

```text
checkpoints/
├── coco/maskedkd_coco.pt
└── waterbirds/maskedkd_waterbirds.pt
```

Both files contain the validation-selected student weights from seed 0. Seeds 1 and 2 are included in the [measured results](../results/README.md); their weights are not published here.

See the [measured results](../results/README.md) and [reference-run protocol](https://github.com/jeehoo0507/kshs-aimlab-benchmarks/tree/codex/maskedkd-reference-runs/reference).

## Official ImageNet initialization

Download the original, **non-distilled** DeiT checkpoints when needed:

```bash
python3 scripts/fetch_deit_imagenet.py tiny small base
```

| Model | Used in current reference runs | Official checkpoint |
| --- | --- | --- |
| DeiT-Tiny | Student | [ImageNet weights](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth) |
| DeiT-Small | Teacher | [ImageNet weights](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) |
| DeiT-Base | Not yet used | [ImageNet weights](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) |

The command requires PyTorch, verifies each download's SHA-256 prefix, and caches the files in PyTorch's hub directory (`~/.cache/torch/hub/checkpoints/` by default). These upstream weights are not committed to this repository. See the [official DeiT model definitions](https://github.com/facebookresearch/deit/blob/main/models.py) for their source.
