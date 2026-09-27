# 결과

[English](../../../results/README.md) · 한국어

CSV에는 COCO single과 Waterbirds에서 측정한 MaskedKD 기준 실험 결과가 들어 있습니다. 데이터셋마다 Teacher 한 번과 Student seed 세 개를 학습했습니다. 학습 코드와 절차는 [기준 실험 브랜치](https://github.com/jeehoo0507/kshs-aimlab-benchmarks/tree/codex/maskedkd-reference-runs/reference)에 있습니다. 이후 다른 방법과 비교할 수 있도록 모든 파일에 `method` 열을 넣었습니다.

결과는 데이터셋별로 [`coco/`](../../../results/coco/)와 [`waterbirds/`](../../../results/waterbirds/)에 나뉘어 있습니다.

| COCO | Waterbirds | 내용 |
| --- | --- | --- |
| [`runs.csv`](../../../results/coco/runs.csv) | [`runs.csv`](../../../results/waterbirds/runs.csv) | Teacher 또는 Student seed당 한 행입니다. Validation으로 선택한 체크포인트와 마지막 epoch의 test 정확도, 학습 비용, 실행 출처를 기록합니다. `test_primary`는 COCO의 macro accuracy 또는 Waterbirds의 worst-group accuracy입니다. |
| [`summary.csv`](../../../results/coco/summary.csv) | [`summary.csv`](../../../results/waterbirds/summary.csv) | Student seed 세 개의 평균과 표본표준편차, 단독 실행한 seed 0의 비용 기준값입니다. |
| [`epochs.csv`](../../../results/coco/epochs.csv) | [`epochs.csv`](../../../results/waterbirds/epochs.csv) | Epoch별 학습 loss와 validation 정확도입니다. |
| [`per_class.csv`](../../../results/coco/per_class.csv) | [`per_group.csv`](../../../results/waterbirds/per_group.csv) | Validation으로 선택한 체크포인트의 test 정답 수·전체 수·정확도입니다. Waterbirds 그룹 번호는 `2*y + place`입니다. |
| [`mask_validation.csv`](../../../results/coco/mask_validation.csv) | [`mask_validation.csv`](../../../results/waterbirds/mask_validation.csv) | Epoch 0, 10, 25, 50, 75, 100의 고정 validation 이미지 전체에서 측정한 Student·전체 Teacher·마스킹 Teacher 정확도, 예측 불일치, KL, 패치 선택 중복도, 전경 패치 재현율·정밀도입니다. 이미지별 마스크는 서버의 `probes.json`에 있으며 `validation_sample_ids.json`과 순서가 맞습니다. |
| [`mask_test.csv`](../../../results/coco/mask_test.csv) | [`mask_test.csv`](../../../results/waterbirds/mask_test.csv) | 최적 체크포인트의 test 진단값입니다. Student·전체 Teacher·마스킹 Teacher 정확도, 예측 불일치, KL, 전경 패치 포함률을 기록합니다. 서버의 `test_mask.json`에는 각 test 이미지 ID, 선택 마스크, 세 모델의 예측이 남아 있습니다. |

정확도는 0~1 사이의 비율입니다. `train_hours`는 최적화 시간만 포함하고, `wall_hours`에는 validation·probe·체크포인트 기록도 포함됩니다. Teacher 행은 196패치를 사용하며 비용은 한 번만 발생합니다. Student seed 0은 비교 가능한 비용 기준을 위해 단독 실행합니다. Seed 1과 2는 병렬 실행할 수 있고 실행 수는 `parallel_jobs`에 기록됩니다.

패치 면적의 절반 이상이 객체 또는 새 마스크 안에 있으면 전경 패치로 봅니다. `foreground_recall`은 전체 전경 패치 중 선택한 비율의 평균이고, `foreground_precision`은 선택한 98패치 중 전경 패치 비율의 평균입니다. 전처리 후 전경 패치가 하나도 없는 이미지는 계산에서 제외하며 그 수는 `foreground_images`로 확인합니다. 전체 Teacher와 마스킹 Teacher는 같은 고정 모델입니다. 예측 불일치는 동일 이미지에 대한 argmax 클래스가 다른지를 비교합니다. `foreground_sha256`은 validation과 test 분석에 사용한 정확한 마스크 버전을 식별합니다.

이미지별 JSON은 서버의 Git 제외 경로인 `outputs/reference/`에 보관하고, 저장소에는 집계 CSV만 넣습니다. JSON의 `selection_hex` 배열은 각 이미지의 196패치 선택 여부를 14×14 행 우선 순서의 비트로 압축한 값이며 끝에 패딩 비트 4개가 있습니다. `1`은 Teacher가 해당 패치를 본다는 뜻입니다. Test 배열의 같은 위치는 `test_mask.json`의 `sample_ids` 및 `predictions`와 대응합니다.

체크포인트는 validation으로만 선택하세요. 실행기는 학습 후 선택한 체크포인트와 마지막 체크포인트를 test에서 평가하지만, test 점수로 방법이나 체크포인트를 고르면 안 됩니다. Student 결과는 데이터셋 해시, Teacher 해시, 초기화, 패치 수, epoch 예산, 전처리, validation 선택 기준이 같을 때만 비교하세요. GPU 비용 비교 시에는 장비와 병렬 실행 수를 맞추거나 단독 seed 0 기록을 사용하세요.
