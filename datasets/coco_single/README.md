# COCO single

![COCO 이미지와 객체 segmentation 예시](../../assets/coco-single.png)

[COCO 2017](https://cocodataset.org/)에서 한 이미지에 주석이 달린 객체들이 하나의 카테고리에만 속하는 이미지를 골라 만든 10-class 분류 데이터셋입니다. 위 이미지는 COCO와 객체 segmentation의 예시이며, 실제 실험 입력은 아래 기준에 따라 별도로 준비합니다.

## Dataset protocol

- **Classes:** `giraffe`, `airplane`, `clock`, `zebra`, `train`, `bird`, `elephant`, `toilet`, `cow`, `bear` 순서로 10개
- **Train / validation:** `train2017` 후보에서 클래스별 최대 600장 / 100장. 후보가 부족하면 모든 클래스에 같은 더 작은 수를 적용
- **Probe:** 클래스별 validation 이미지 20장
- **Test:** `val2017`에서 조건을 만족하는 이미지를 별도로 사용
- **Split seed:** `20260922`

같은 카테고리의 객체가 한 이미지에 여러 개 있는 것은 허용합니다. crowd, 면적 0 이하, segmentation 누락 주석은 제외합니다. 중복 이미지는 디코딩된 RGB 해시가 같거나 64-bit dHash 거리가 4 이하인 경우 제거합니다. FG mask는 이미지에 속한 모든 객체의 segmentation을 합칩니다.

이미지는 224×224로 resize하며, 학습할 때만 좌우 반전을 적용합니다. 16×16 patch 기준 공간 토큰은 196개입니다. 이미지 선정·분할 코드는 데이터를 생성한 실험 구현과 함께 보존해야 합니다.

## Data layout

```text
coco_single/
  manifest.json
  images/train2017/*.jpg
  images/val2017/*.jpg
  masks/train2017/*.png
  masks/val2017/*.png
```

`manifest.json`에는 클래스 순서와 각 이미지의 `id`, `label`, `split`, `probe`, 이미지·mask 상대경로 및 SHA-256을 기록합니다. 실제 클래스별 수와 test 수는 생성된 manifest에서 확인합니다. 이 저장소에는 원본 데이터와 생성 스크립트가 들어 있지 않습니다.

## Validation

저장소 루트에서 실행합니다.

```bash
python3 scripts/check_assets.py coco /absolute/path/to/coco_single
python3 scripts/check_assets.py coco /absolute/path/to/coco_single --full
```

첫 명령은 manifest 구조와 파일 존재 여부를, `--full`은 이미지·mask 해시까지 확인합니다.
