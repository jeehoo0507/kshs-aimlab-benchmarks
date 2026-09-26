# Waterbirds

<p align="center">
  <img src="../../assets/waterbirds.png" alt="새 종류와 배경 조합으로 나뉜 Waterbirds의 네 그룹 예시" width="595">
</p>

[Waterbirds](https://github.com/kohpangwei/group_DRO#waterbirds)는 CUB 새 이미지를 Places 배경에 합성한 데이터셋입니다. 새 종류 `y`(landbird/waterbird)와 배경 `place`(land/water)의 조합으로 네 그룹을 나눕니다. 위 그림은 그룹의 예시이며, 실제 샘플 수는 준비한 `metadata.csv`로 확인합니다.

## 평가 구성

원본 `metadata.csv`의 `split` 값 0/1/2를 각각 train/validation/test로 사용합니다. 학습 데이터에서 새 종류와 배경은 강하게 연관되어 있습니다. 평가에서는 전체 정확도와 함께 **그룹별 정확도 및 worst-group accuracy**를 확인합니다.

기본 분류 평가에는 합성 이미지와 `metadata.csv`가 필요합니다. FG/BG patch 분석을 하려면 원본 CUB segmentation PNG도 필요합니다. Segmentation이 없는 데이터는 FG/BG 실험에 사용하지 않습니다.

## 파일 구성

```text
waterbird_complete95_forest2water2/
  metadata.csv
  <metadata.csv의 img_filename 경로에 해당하는 이미지>
CUB_200_2011/segmentations/
  <각 이미지 상대경로와 이름이 대응하는 PNG mask>
```

기존 token-budget 실험은 학습 시 이미지와 mask에 동일한 random resized crop(224×224) 및 좌우 반전을 적용했습니다. 평가는 resize 256 후 center crop 224를 사용했습니다. 다른 전처리를 사용했다면 결과에 별도 조건으로 기록합니다.

## 데이터 검증

저장소 루트에서 실행합니다. 두 번째 명령은 FG/BG 분석용 mask까지 확인할 때 사용합니다.

```bash
python3 scripts/check_assets.py waterbirds /absolute/path/to/waterbird_complete95_forest2water2
python3 scripts/check_assets.py waterbirds /absolute/path/to/waterbird_complete95_forest2water2 --seg-root /absolute/path/to/CUB_200_2011/segmentations
```
