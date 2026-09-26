# Waterbirds

<p align="center">
  <img src="../../assets/waterbirds.png" alt="새 종류와 배경 조합으로 나뉜 Waterbirds의 네 그룹 예시" width="595">
</p>

Waterbirds의 새 종류 × 배경 조합을 보여 주는 예시입니다. 그림에 적힌 샘플 수는 실제 준비한 데이터의 수를 대신하지 않으므로 `metadata.csv`에서 확인합니다.

[원본 Waterbirds](https://github.com/kohpangwei/group_DRO#waterbirds)는 CUB 새 이미지를 Places 배경에 합성한 데이터입니다. 새 라벨 `y`(landbird/waterbird)와 배경 `place`(land/water)를 조합한 **4개 그룹**으로 봅니다. 원본 `metadata.csv`의 `split` 0/1/2를 train/validation/test로 사용합니다. Train은 배경과 라벨의 상관관계가 강하고, validation/test는 그룹 비율이 다르므로 평가 시 그룹별 정확도와 worst-group accuracy를 함께 보고합니다.

기본 분류 평가에는 합성 이미지와 `metadata.csv`가 필요합니다. FG/BG patch 분석에는 원본 CUB segmentation PNG도 별도로 필요합니다. Segmentation이 없는 Waterbirds 사본을 baseline용으로 사용할 수는 있지만 FG/BG 실험 입력으로 표시하면 안 됩니다.

```text
waterbird_complete95_forest2water2/
  metadata.csv
  <metadata.csv의 img_filename 경로에 해당하는 이미지>
CUB_200_2011/segmentations/
  <각 이미지 상대경로와 이름이 대응하는 PNG mask>
```

기존 token-budget 실험의 train 전처리는 이미지와 mask에 같은 random resized crop(224×224) 및 좌우 반전을 적용합니다. 평가는 resize 256 후 center crop 224입니다. 다른 전처리를 사용하면 별도 조건으로 기록합니다.

```bash
python3 scripts/check_assets.py waterbirds /absolute/path/to/waterbird_complete95_forest2water2
python3 scripts/check_assets.py waterbirds /absolute/path/to/waterbird_complete95_forest2water2 --seg-root /absolute/path/to/CUB_200_2011/segmentations
```

첫 명령은 metadata, 이미지, 분할·그룹을 확인합니다. 두 번째는 FG/BG 분석용 mask 존재 여부도 확인합니다.
