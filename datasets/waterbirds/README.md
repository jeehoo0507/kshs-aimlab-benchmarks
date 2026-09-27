# Waterbirds

English · [한국어](../../docs/ko/datasets/waterbirds/README.md)

<p align="center">
  <img src="../../assets/waterbirds.png" alt="Waterbirds groups by bird type and background" width="595">
</p>

[Waterbirds](https://github.com/kohpangwei/group_DRO#waterbirds) places CUB bird images on Places backgrounds. Bird type `y` (landbird/waterbird) and background `place` (land/water) define four groups. The figure illustrates these groups; check your `metadata.csv` for actual counts.

## Protocol

- Use the original `metadata.csv` splits: `0` train, `1` validation, `2` test.
- Report overall, per-group, and worst-group accuracy. Training labels and backgrounds are strongly correlated.
- Classification requires the composite images and metadata. FG/BG patch analysis also requires the original CUB segmentation masks.
- Prior token-budget runs used a shared random resized crop (224×224) and horizontal flip for training images and masks; evaluation used resize 256 followed by center crop 224. Record any preprocessing changes.

## Files

```text
waterbird_complete95_forest2water2/
  metadata.csv
  <images listed in metadata.csv>
CUB_200_2011/segmentations/
  <matching PNG masks, for FG/BG analysis>
```

## Download

From the repository root:

```bash
bash datasets/waterbirds/setup.sh
```

This downloads the [original group_DRO Waterbirds release](https://github.com/kohpangwei/group_DRO#waterbirds) to the Git-ignored `data/waterbird_complete95_forest2water2/` directory and validates the images and metadata. It preserves the published train/validation/test split. CUB segmentation masks are separate and are not included in this download.

For foreground patch coverage in reference runs, download the [official CUB segmentations](https://data.caltech.edu/records/w9d68-gec53):

```bash
python3 datasets/waterbirds/download_masks.py
```

The reference runner executes this automatically when the masks are absent. It verifies the archive checksum and matches every Waterbirds image to its CUB mask.

## Validation

```bash
python3 scripts/check_assets.py waterbirds data/waterbird_complete95_forest2water2
python3 scripts/check_assets.py waterbirds data/waterbird_complete95_forest2water2 --seg-root /path/to/CUB_200_2011/segmentations
```
