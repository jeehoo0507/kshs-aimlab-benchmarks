# Datasets

Dataset images and masks are not stored in Git. Run the setup commands from the repository root:

```bash
bash datasets/coco_single/setup.sh
bash datasets/waterbirds/setup.sh
```

Each command downloads data into `data/`, prepares the required structure, and validates it.

| Dataset | Setup result | Guide |
| --- | --- | --- |
| COCO single | Download COCO 2017 annotations and eligible images; generate masks and splits | [COCO single](coco_single/README.md) |
| Waterbirds | Download the original group_DRO release; keep its provided splits | [Waterbirds](waterbirds/README.md) |

Record the SHA-256 of the COCO manifest or Waterbirds metadata with every result to identify the exact split.
