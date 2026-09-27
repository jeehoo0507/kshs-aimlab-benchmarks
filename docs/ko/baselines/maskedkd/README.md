# MaskedKD 기준 구현

[English](../../../../baselines/maskedkd/README.md) · 한국어

비교 기준은 [MaskedKD 논문](https://arxiv.org/abs/2302.10494)과 commit `96d052da7346441e2425876a9ad37ff8c87fe383`에 고정한 [공식 구현](https://github.com/effl-lab/MaskedKD)입니다.

```bash
bash scripts/fetch_maskedkd.sh
```

이 명령은 원본 코드를 Git에서 제외된 `external/MaskedKD/`에 클론합니다. 직접 학습한 비교용 가중치는 [체크포인트](../../checkpoints/README.md)에서 확인할 수 있습니다. 결과마다 코드 commit과 데이터셋 manifest 또는 metadata 해시를 기록하세요.
