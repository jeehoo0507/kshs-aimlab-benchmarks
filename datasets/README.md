# Datasets

Dataset images and masks are not stored in this repository. Prepare them locally, then use the validation commands in each guide.

| Dataset | Required files | Guide |
| --- | --- | --- |
| COCO single | `manifest.json`, images, masks | [COCO single](coco_single/README.md) |
| Waterbirds | Images, `metadata.csv`; CUB masks for FG/BG analysis | [Waterbirds](waterbirds/README.md) |

Record the SHA-256 of the COCO manifest or Waterbirds metadata with every result to identify the exact split.
