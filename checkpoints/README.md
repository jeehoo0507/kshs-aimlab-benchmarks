# 체크포인트

가중치 파일은 팀 공유 저장 공간에 둡니다. `index.csv`에는 실제로 확인한 파일만 한 줄씩 등록합니다. `relative_path`는 점검할 때 지정하는 `--root` 아래의 경로입니다. `sha256`은 파일 내용의 해시이며, 파일을 바꾸면 새 행으로 등록합니다.

필드: `id`, `dataset`, `architecture`, `role`(teacher/student), `seed`, `source`, `relative_path`, `sha256`.

```bash
python3 scripts/check_assets.py checkpoints --root /absolute/path/to/shared/checkpoints --full
```

아직 등록된 체크포인트가 없으면 표에는 헤더만 있습니다. 기존 실험의 `.pt` 파일을 임의로 이 저장소에 복사해 커밋하지 않습니다.

