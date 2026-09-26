# COCO single

![COCO instance segmentation examples](../../assets/coco-single.png)

A 10-class classification subset of [COCO 2017](https://cocodataset.org/). Each image contains annotations from one category; multiple instances of that category are allowed.

## Protocol

- **Classes:** `giraffe`, `airplane`, `clock`, `zebra`, `train`, `bird`, `elephant`, `toilet`, `cow`, `bear` (in this order).
- **Train / validation:** Up to 600 / 100 images per class from `train2017`. If a class has fewer candidates, use the same lower count for every class.
- **Probe:** 20 validation images per class. **Test:** eligible images from `val2017`. **Split seed:** `20260922`.
- Exclude crowd, non-positive-area, and missing-segmentation annotations. Remove duplicates with identical decoded RGB hashes or a 64-bit dHash distance of at most 4.
- Union all instance segmentations into one foreground mask. Resize images to 224×224; apply horizontal flips only during training. A 16×16 patch size gives 196 spatial tokens.

## Files

```text
coco_single/
  manifest.json
  images/{train2017,val2017}/*.jpg
  masks/{train2017,val2017}/*.png
```

The manifest records the class order and each image's `id`, `label`, `split`, `probe` flag, relative image/mask paths, and SHA-256 hashes. Source images and the dataset-generation script are not included here.

## Validation

```bash
python3 scripts/check_assets.py coco /path/to/coco_single
python3 scripts/check_assets.py coco /path/to/coco_single --full
```

The first command checks the manifest and file presence; `--full` also checks file hashes.
