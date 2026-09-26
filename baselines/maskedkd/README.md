# MaskedKD baseline

기준은 [MaskedKD 논문](https://arxiv.org/abs/2302.10494)과 [공식 구현](https://github.com/effl-lab/MaskedKD)입니다. 공식 저장소의 commit `96d052da7346441e2425876a9ad37ff8c87fe383`을 기준으로 고정합니다.

```bash
bash scripts/fetch_maskedkd.sh
```

명령은 공식 코드를 `external/MaskedKD/`에 가져옵니다. 이 경로는 Git에서 제외합니다. 저자가 직접 학습한 baseline 가중치는 [`checkpoints/`](../../checkpoints/README.md)에 등록합니다. 공식 구현의 전처리와 COCO·Waterbirds 실험의 전처리가 다를 수 있으므로, 결과를 비교할 때 코드 commit과 데이터 manifest 또는 metadata 해시를 함께 기록합니다.
