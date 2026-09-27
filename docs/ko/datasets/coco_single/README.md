# COCO single

[English](../../../../datasets/coco_single/README.md) · 한국어

![COCO 인스턴스 분할 예시](../../../../assets/coco-single.png)

[COCO 2017](https://cocodataset.org/)에서 구성한 10개 클래스 분류 데이터셋입니다. 선택된 각 이미지에는 주석이 있는 객체 인스턴스가 정확히 하나 있습니다.

## 구성 기준

- **클래스:** `giraffe`, `airplane`, `clock`, `zebra`, `train`, `bird`, `elephant`, `toilet`, `stop sign`, `bear` 순서입니다.
- **Train / validation:** `train2017`에서 클래스별 균형을 맞춥니다. 검증된 split은 클래스마다 train 321장, validation 53장입니다.
- **Probe:** 클래스마다 validation 20장입니다. **Test:** `val2017`에서 조건에 맞는 297장입니다. **Split seed:** `20260922`입니다.
- 주석 객체가 없거나 둘 이상인 이미지, crowd 주석, 면적이 0 이하인 주석, segmentation이 없는 이미지는 제외합니다. 디코딩한 RGB 해시가 같거나 64비트 dHash의 해밍 거리가 4 이하인 중복 이미지도 제거합니다.
- 객체 segmentation을 전경 마스크로 사용합니다. 이미지는 224×224로 조정하며 가로 뒤집기는 학습 중에만 적용합니다. 16×16 패치 기준 공간 토큰은 196개입니다.

## 파일 구조

```text
coco_single/
  manifest.json
  images/{train2017,val2017}/*.jpg
  masks/{train2017,val2017}/*.png
```

Manifest에는 단일 객체 선택 기준, 클래스 순서와 각 이미지의 `id`, `label`, `split`, `probe` 여부, 이미지·마스크 경로 및 SHA-256이 기록됩니다. 다운로드한 데이터는 Git에서 제외된 `data/coco_single/`에 저장됩니다.

## 준비

저장소 루트에서 실행하세요.

```bash
bash datasets/coco_single/setup.sh
```

이 명령은 `.venv-data/`에 준비용 의존성을 설치하고, COCO 2017 주석과 조건에 맞는 원본 이미지를 다운로드합니다. 이어서 전경 마스크 생성, 중복 제거, train/validation/test 분할 및 검증을 수행합니다. GPU는 사용하지 않습니다. 다시 실행하면 완료된 데이터를 재사용하거나 다운로드를 이어갑니다. `data/coco_single/`에 이전 버전의 데이터가 있다면 다른 위치로 옮긴 뒤 실행하세요. 이전 클래스 구성으로 만든 모델과 결과는 직접 비교할 수 없습니다. 준비 코드는 [aim-lab-test-2](https://github.com/jeehoo0507/aim-lab-test-2)의 commit `0ee46e4`를 바탕으로 했으며, 클래스와 단일 인스턴스 기준은 이 저장소에서 정의합니다.

## 검증

```bash
python3 scripts/check_assets.py coco data/coco_single
python3 scripts/check_assets.py coco data/coco_single --full
```

첫 명령은 manifest와 파일 존재 여부를 확인하고, `--full`은 각 파일의 해시까지 확인합니다.
