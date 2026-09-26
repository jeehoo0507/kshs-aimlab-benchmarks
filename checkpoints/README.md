# MaskedKD checkpoints

이 디렉터리에는 저자가 **직접 학습한 MaskedKD baseline checkpoint**를 올립니다. 현재 `index.csv`에는 헤더만 있으며, 공개된 가중치 파일은 없습니다.

## File layout

Checkpoint를 추가할 때는 dataset별로 파일을 정리합니다.

```text
checkpoints/
  index.csv
  maskedkd/
    coco_single/
    waterbirds/
```

`index.csv`의 각 행은 실제 checkpoint 파일 하나에 대응합니다. `relative_path`는 `checkpoints/`를 기준으로 한 경로입니다. `id`, `dataset`, `architecture`, `role`(teacher/student), `seed`, `source`(원본 실험 실행 기록), `relative_path`, `sha256`을 기록합니다. 가중치 파일과 metadata를 확인한 뒤 등록합니다.

## Verify

저장소 루트에서 checkpoint 파일의 존재 여부와 SHA-256을 확인합니다.

```bash
python3 scripts/check_assets.py checkpoints --full
```

다른 위치에서 파일을 확인해야 할 때만 `--root /absolute/path/to/checkpoints`를 지정합니다. 등록된 파일이 없으면 검사 결과에 `0 registered`가 표시됩니다.
