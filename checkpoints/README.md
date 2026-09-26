# Checkpoints

MaskedKD checkpoints trained by the author for baseline comparison.

```text
checkpoints/
├── coco/maskedkd_seed0.pt
└── waterbirds/maskedkd_seed0.pt
```

These files will be created by the [reference runner](../reference/README.md) after training. The frozen teacher checkpoints remain in Git-ignored `outputs/reference/` and must be retained for exact future comparisons.
