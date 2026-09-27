# Datasets

Dataset images and masks are not stored in Git. Run the setup commands from the repository root:

```bash
bash datasets/coco_single/setup.sh
bash datasets/waterbirds/setup.sh
bash datasets/imagenet/setup.sh /path/to/official-archives
```

COCO single and Waterbirds download their source data. ImageNet requires the three official train, validation, and devkit archives to be downloaded with your ImageNet account first. Setup commands prepare data in `data/` by default and validate it.

| Dataset | Setup result | Guide |
| --- | --- | --- |
| COCO single | Select one annotated object per image from 10 classes; generate masks and splits | [COCO single](coco_single/README.md) |
| Waterbirds | Download the original group_DRO release; keep its provided splits | [Waterbirds](waterbirds/README.md) |
| ImageNet-1K | Prepare labeled train and validation splits from official archives; no test split | [ImageNet-1K](imagenet/README.md) |

Record the SHA-256 of the COCO or ImageNet manifest, or Waterbirds metadata, with every result to identify the exact split.
