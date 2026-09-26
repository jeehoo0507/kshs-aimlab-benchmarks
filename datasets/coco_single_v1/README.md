# COCO single v1

[COCO 2017](https://cocodataset.org/)의 instance annotation에서 한 이미지의 주석이 **하나의 카테고리**에만 속하는 후보를 고릅니다. 한 이미지에 같은 카테고리의 객체가 여러 개 있는 것은 허용합니다. crowd, 면적 0 이하, segmentation 누락은 제외합니다.

클래스는 `giraffe, airplane, clock, zebra, train, bird, elephant, toilet, cow, bear`의 10개입니다. `train2017` 후보에서 클래스별 최대 600장을 train, 최대 100장을 validation에 사용하며, 후보가 부족하면 모든 클래스에 같은 더 작은 수를 적용합니다. 클래스별 validation 20장은 probe에 포함됩니다. `val2017`에서 조건을 만족하는 이미지는 held-out test로 사용합니다. 분할 seed는 `20260922`입니다.

중복 제거는 디코딩된 RGB 해시 일치 또는 64-bit dHash 거리 4 이하 기준을 사용합니다. FG mask는 해당 이미지의 모든 객체 segmentation을 합친 것입니다. 전처리는 전체 이미지를 224×224로 resize하고 train에서만 좌우 반전합니다. 16×16 patch 기준 공간 토큰은 196개입니다. 자세한 선택·분할 코드는 이 버전을 만든 실험 구현과 함께 보존해야 합니다.

예상 파일 배치:

```text
coco_single/
  manifest.json
  images/train2017/*.jpg
  images/val2017/*.jpg
  masks/train2017/*.png
  masks/val2017/*.png
```

`manifest.json`에는 클래스 순서, 각 이미지의 `id`, `label`, `split`, `probe`, 이미지·mask 상대경로와 SHA-256이 있어야 합니다. 생성된 실제 클래스별 수와 test 수는 manifest에서 확인합니다. 원본 이미지는 Git에 넣지 않습니다.

```bash
python3 scripts/check_assets.py coco /absolute/path/to/coco_single
python3 scripts/check_assets.py coco /absolute/path/to/coco_single --full
```

첫 명령은 구조와 파일 존재 여부를, `--full`은 이미지·mask 해시까지 확인합니다.

