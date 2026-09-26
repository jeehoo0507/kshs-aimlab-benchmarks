# 비교 결과

`coco.csv`와 `waterbirds.csv`는 같은 열을 사용합니다. 한 행은 **하나의 seed·평가 checkpoint·지표**입니다. 여러 seed의 평균을 개별 실행 결과처럼 적지 않습니다.

- `dataset_version`: 데이터 정의 버전 (`coco_single_v1`, `waterbirds_v1`)
- `method`, `teacher_id`, `student_arch`, `student_init`, `keep_patches`, `seed`: 실험 조건
- `eval_split`, `checkpoint`, `metric`, `value`: 평가 조건과 결과
- `train_hours`, `peak_vram_gib`: 측정된 비용; 측정하지 않았으면 빈칸
- `code_commit`, `run_ref`: 재현에 필요한 코드 revision과 원본 실행 위치

COCO는 macro accuracy, Waterbirds는 worst-group accuracy를 우선 기록하고 필요한 보조 지표를 추가 행으로 적습니다. 비교표를 만들 때 같은 데이터 버전, teacher, 초기화, patch 수, seed와 checkpoint 선택 기준을 맞춥니다. 검증되지 않은 숫자는 넣지 않습니다.

