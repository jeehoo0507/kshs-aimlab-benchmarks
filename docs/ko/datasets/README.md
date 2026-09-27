# 데이터셋

[English](../../../datasets/README.md) · 한국어

데이터셋 이미지와 마스크는 Git에 저장하지 않습니다. 저장소 루트에서 다음 명령을 실행하세요.

```bash
bash datasets/coco_single/setup.sh
bash datasets/waterbirds/setup.sh
bash datasets/imagenet/setup.sh
```

COCO single과 Waterbirds는 원본 데이터를 자동으로 다운로드합니다. ImageNet은 계정으로 공식 train·validation·devkit 아카이브 3개를 받아 `data/imagenet/archives/`에 먼저 넣어야 합니다. 준비 명령은 기본적으로 `data/`에 데이터를 구성한 뒤 검증합니다.

| 데이터셋 | 준비 결과 | 안내 |
| --- | --- | --- |
| COCO single | 10개 클래스에서 이미지당 주석 객체가 하나인 샘플을 고르고 마스크와 split 생성 | [COCO single](coco_single/README.md) |
| Waterbirds | 원본 group_DRO 데이터를 받고 제공된 split 유지 | [Waterbirds](waterbirds/README.md) |
| ImageNet-1K | 공식 아카이브에서 라벨이 있는 train·validation 구성; test는 제외 | [ImageNet-1K](imagenet/README.md) |

결과를 기록할 때는 정확한 split을 식별할 수 있도록 COCO·ImageNet manifest 또는 Waterbirds metadata의 SHA-256을 함께 남기세요.
