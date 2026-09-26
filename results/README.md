# Results

The CSVs are empty templates until the MaskedKD reference runs finish. Run the [reference protocol](../reference/README.md) to fill them. Each file includes a `method` column so later methods can be compared against `maskedkd` under the same conditions.

| File | Contents |
| --- | --- |
| [`coco.csv`](coco.csv), [`waterbirds.csv`](waterbirds.csv) | One row per completed teacher or student seed: validation-selected and last-epoch test accuracy, training cost, provenance. `test_primary` is COCO macro accuracy or Waterbirds worst-group accuracy. |
| [`coco_summary.csv`](coco_summary.csv), [`waterbirds_summary.csv`](waterbirds_summary.csv) | Three-seed mean and sample standard deviation once all student seeds finish; isolated seed-0 cost reference. |
| [`coco_epochs.csv`](coco_epochs.csv), [`waterbirds_epochs.csv`](waterbirds_epochs.csv) | Training loss and validation accuracy at every epoch. |
| [`coco_per_class.csv`](coco_per_class.csv), [`waterbirds_per_group.csv`](waterbirds_per_group.csv) | Test correct/total counts and accuracy for the validation-selected checkpoint. Waterbirds group is `2*y + place`. |
| [`coco_mask_probe.csv`](coco_mask_probe.csv), [`waterbirds_mask_probe.csv`](waterbirds_mask_probe.csv) | Diagnostics on **every fixed validation image** at epochs 0, 10, 25, 50, 75, 100: student/full-teacher/masked-teacher accuracy, prediction disagreement, KL, patch-selection overlap, and foreground patch recall/precision for both datasets. Per-image selected masks remain in `probes.json`, aligned with `validation_sample_ids.json`. |
| [`coco_mask_test.csv`](coco_mask_test.csv), [`waterbirds_mask_test.csv`](waterbirds_mask_test.csv) | Final best-checkpoint test diagnostics: student/full-teacher/masked-teacher accuracy, prediction disagreement, KL, and foreground patch coverage. The run's `test_mask.json` retains each test sample ID, selected mask, and three predictions. |

Accuracy values are fractions from 0 to 1. `train_hours` counts optimization only; `wall_hours` also includes validation, probes, and checkpoint writes. The teacher row has 196 patches and is a separate one-time cost. Student seed 0 runs alone to provide a comparable cost reference; seeds 1 and 2 may run in parallel, identified by `parallel_jobs`.

A patch counts as foreground when at least half its pixels lie inside the object/bird mask. `foreground_recall` is the mean fraction of foreground patches retained, and `foreground_precision` is the mean fraction of the 98 selected patches that are foreground; images with no foreground patch after preprocessing are excluded and counted by `foreground_images`. The full and masked teacher are the same frozen model. Prediction disagreement compares class argmax outputs on the same images. `foreground_sha256` identifies the exact validation and test masks used.

The `selection_hex` arrays encode each image's 196 selected/not-selected patches as packed bits in row-major 14×14 patch order, with four padding bits at the end. A value of `1` means the teacher sees that patch. The test arrays are aligned by index with `sample_ids` and `predictions` in `test_mask.json`.

Choose checkpoints using validation only. The runner evaluates the selected and final checkpoints on test after training; never select a method or checkpoint from test scores. Compare student rows only when dataset hash, teacher hash, initialization, patch count, epoch budget, preprocessing, and validation-selection rule match. For GPU cost comparisons, match hardware and concurrency or use the isolated seed-0 reference.
