# 데이터셋

이 디렉터리는 실험에 사용할 데이터의 **선정 기준과 검증 방법**을 설명합니다. 원본 이미지와 segmentation mask는 포함하지 않습니다. 로컬 `data/` 또는 팀 공유 저장 공간에 데이터를 준비한 뒤 아래 문서와 대조하세요.

| 데이터셋 | 필요한 입력 | 자세한 설명 |
| --- | --- | --- |
| COCO single | COCO 2017에서 만든 `manifest.json`, 이미지, mask | [COCO single](coco_single/README.md) |
| Waterbirds | 합성 이미지와 `metadata.csv`; FG/BG 분석에는 CUB segmentation | [Waterbirds](waterbirds/README.md) |

COCO single의 `manifest.json`은 이미지별 라벨·분할·파일 해시를 기록합니다. Waterbirds는 원본 `metadata.csv`의 분할을 따릅니다. 결과를 기록할 때는 사용한 `manifest.json` 또는 `metadata.csv`의 SHA-256을 함께 남겨 데이터 구성을 식별합니다.
