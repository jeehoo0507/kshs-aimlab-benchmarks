# Results

The CSVs are empty templates until the MaskedKD reference runs finish. Run the [reference protocol](../reference/README.md) to fill them. Each file includes a `method` column so later methods can be compared against `maskedkd` under the same conditions.

| File | Contents |
| --- | --- |
| [`coco.csv`](coco.csv), [`waterbirds.csv`](waterbirds.csv) | One row per completed teacher or student seed: validation-selected and last-epoch test accuracy, training cost, provenance. `test_primary` is COCO macro accuracy or Waterbirds worst-group accuracy. |
| [`coco_summary.csv`](coco_summary.csv), [`waterbirds_summary.csv`](waterbirds_summary.csv) | Three-seed mean and sample standard deviation once all student seeds finish; isolated seed-0 cost reference. |
| [`coco_epochs.csv`](coco_epochs.csv), [`waterbirds_epochs.csv`](waterbirds_epochs.csv) | Training loss and validation accuracy at every epoch. |
| [`coco_per_class.csv`](coco_per_class.csv), [`waterbirds_per_group.csv`](waterbirds_per_group.csv) | Test correct/total counts and accuracy for the validation-selected checkpoint. Waterbirds group is `2*y + place`. |
| [`coco_mask_probe.csv`](coco_mask_probe.csv), [`waterbirds_mask_probe.csv`](waterbirds_mask_probe.csv) | Fixed validation-probe diagnostics at epochs 0, 10, 25, 50, 75, 100: full/masked teacher accuracy, KL, selection overlap with the initial and previous probe, and COCO foreground coverage. Waterbirds foreground fields are empty because the standard download has no segmentation masks. |

Accuracy values are fractions from 0 to 1. `train_hours` counts optimization only; `wall_hours` also includes validation, probes, and checkpoint writes. The teacher row has 196 patches and is a separate one-time cost. Student seed 0 runs alone to provide a comparable cost reference; seeds 1 and 2 may run in parallel, identified by `parallel_jobs`.

Choose checkpoints using validation only. The runner evaluates the selected and final checkpoints on test after training; never select a method or checkpoint from test scores. Compare student rows only when dataset hash, teacher hash, initialization, patch count, epoch budget, preprocessing, and validation-selection rule match. For GPU cost comparisons, match hardware and concurrency or use the isolated seed-0 reference.
