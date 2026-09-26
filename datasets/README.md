# 데이터셋

이 저장소에는 데이터의 **정의와 확인 도구**를 둡니다. 원본 이미지와 segmentation은 `data/` 또는 팀 공유 저장 공간에 두며 Git에 추가하지 않습니다. 기존에 만든 데이터를 재사용할 때도 문서의 기준과 실제 파일을 대조하세요.

| 데이터셋 | 용도 | 자세한 구성 |
| --- | --- | --- |
| COCO single v1 | 10-class 분류 및 patch 선택 분석 | [coco_single_v1](coco_single_v1/README.md) |
| Waterbirds v1 | 새 종류 × 배경의 허위 상관관계 평가 | [waterbirds_v1](waterbirds_v1/README.md) |

`manifest.json`은 어떤 이미지가 어떤 라벨과 분할에 속하는지 적은 목록입니다. 같은 이름의 데이터셋이라도 이 목록이 다르면 별도 버전으로 취급합니다.

