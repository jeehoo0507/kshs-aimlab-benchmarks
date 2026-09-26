"""Export completed reference runs into reviewable, method-compatible CSVs."""
import csv
import json
import os
import shutil
import statistics
from pathlib import Path

from .engine import ROOT, run_directory, sha256


MAIN_FIELDS = ("dataset", "data_sha256", "method", "teacher_sha256", "pretrained_sha256",
               "student_arch", "student_init",
               "keep_patches", "seed", "epochs", "selection_metric", "best_epoch", "val_score",
               "test_primary", "test_overall", "last_test_primary", "last_test_overall",
               "train_hours", "wall_hours", "peak_vram_gib",
               "samples_per_second", "gpu", "parallel_jobs", "checkpoint_sha256", "code_sha256",
               "code_commit", "run_ref")
EPOCH_FIELDS = ("dataset", "method", "seed", "epoch", "lr", "train_loss", "train_ce", "train_kd",
                "train_accuracy", "val_primary", "val_overall", "train_seconds", "epoch_seconds")
CLASS_FIELDS = ("dataset", "method", "seed", "class_index", "class_name", "correct", "n", "accuracy")
GROUP_FIELDS = ("dataset", "method", "seed", "group", "label", "background", "correct", "n", "accuracy")
PROBE_FIELDS = ("dataset", "method", "seed", "epoch", "n", "teacher_full_accuracy",
                "teacher_masked_accuracy", "full_correct_masked_wrong", "kl_full_to_masked",
                "foreground_recall", "foreground_precision", "foreground_images",
                "selection_overlap_initial", "selection_overlap_previous")
SUMMARY_FIELDS = ("dataset", "method", "data_sha256", "teacher_sha256", "seeds",
                  "test_primary_mean", "test_primary_sd", "test_overall_mean", "test_overall_sd",
                  "isolated_seed0_train_hours", "isolated_seed0_peak_vram_gib",
                  "isolated_seed0_samples_per_second")


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep future methods in the shared CSV; only replace the reference rows.
    if path.exists():
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            old = list(reader)
            if old and tuple(reader.fieldnames or ()) != fields:
                raise ValueError(f"Cannot merge incompatible CSV schema: {path}")
        rows = [row for row in old if row.get("method") not in ("teacher", "maskedkd")] + rows
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def completed_runs(output_root, dataset, seeds):
    for role, seed in [("teacher", 0), *(("student", s) for s in seeds)]:
        directory = run_directory(output_root, dataset, role, seed)
        result = directory / "result.json"
        if result.exists():
            yield directory, json.loads(result.read_text())


def export(output_root, dataset, seeds=(0, 1, 2), publish_checkpoint=True):
    output_root = Path(output_root)
    main, epochs, classes, groups, probes = [], [], [], [], []
    for directory, result in completed_runs(output_root, dataset, seeds):
        method, seed = result["method"], result["seed"]
        test = result["best_test"]
        metric = result["selection_metric"]
        count = result["epochs"] * result["train_samples"]
        main.append({"dataset": dataset, "data_sha256": result["dataset_sha256"], "method": method,
                     "teacher_sha256": result["teacher_sha256"],
                     "pretrained_sha256": result["pretrained_sha256"], "student_arch": result["student_arch"],
                     "student_init": result["student_init"],
                     "keep_patches": result["config"]["keep_patches"] if method == "maskedkd" else 196,
                     "seed": seed, "epochs": result["epochs"], "selection_metric": metric,
                     "best_epoch": result["best_epoch"], "val_score": result["best_validation"],
                     "test_primary": test[metric], "test_overall": test["overall_accuracy"],
                     "last_test_primary": result["last_test"][metric],
                     "last_test_overall": result["last_test"]["overall_accuracy"],
                     "train_hours": result["train_seconds"] / 3600,
                     "wall_hours": result["wall_seconds"] / 3600,
                     "peak_vram_gib": result["peak_vram_gib"],
                     "samples_per_second": count / result["train_seconds"],
                     "gpu": result["environment"]["gpu"],
                     "parallel_jobs": result["parallel_jobs"],
                     "checkpoint_sha256": result["checkpoint_sha256"],
                     "code_sha256": result["code_sha256"],
                     "code_commit": result["code_commit"],
                     "run_ref": str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory)})
        history = json.loads((directory / "history.json").read_text())
        for row in history:
            epochs.append({"dataset": dataset, "method": method, "seed": seed,
                           "epoch": row["epoch"], "lr": row["lr"],
                           "train_loss": row["train"]["loss"], "train_ce": row["train"]["ce"],
                           "train_kd": row["train"]["kd"], "train_accuracy": row["train"]["accuracy"],
                           "val_primary": row["validation"][metric],
                           "val_overall": row["validation"]["overall_accuracy"],
                           "train_seconds": row["train"]["seconds"], "epoch_seconds": row["epoch_seconds"]})
        if dataset == "coco":
            for index, (correct, n, accuracy) in enumerate(zip(test["class_correct"], test["class_counts"],
                                                               test["class_accuracy"])):
                classes.append({"dataset": dataset, "method": method, "seed": seed,
                                "class_index": index, "class_name": result["class_names"][index], "correct": correct,
                                "n": n, "accuracy": accuracy})
        else:
            for group, (correct, n, accuracy) in enumerate(zip(test["group_correct"], test["group_counts"],
                                                                test["group_accuracy"])):
                groups.append({"dataset": dataset, "method": method, "seed": seed, "group": group,
                               "label": group // 2, "background": group % 2,
                               "correct": correct, "n": n, "accuracy": accuracy})
        if method == "maskedkd":
            for row in json.loads((directory / "probes.json").read_text()):
                full = {"dataset": dataset, "method": method, "seed": seed, **row}
                probes.append({field: full.get(field) for field in PROBE_FIELDS})
    destination = ROOT / "results"
    write_csv(destination / f"{dataset}.csv", MAIN_FIELDS, main)
    write_csv(destination / f"{dataset}_epochs.csv", EPOCH_FIELDS, epochs)
    write_csv(destination / ("coco_per_class.csv" if dataset == "coco" else "waterbirds_per_group.csv"),
              CLASS_FIELDS if dataset == "coco" else GROUP_FIELDS, classes if dataset == "coco" else groups)
    write_csv(destination / f"{dataset}_mask_probe.csv", PROBE_FIELDS, probes)
    students = sorted((row for row in main if row["method"] == "maskedkd"), key=lambda row: row["seed"])
    summary = []
    if len(students) == 3 and [row["seed"] for row in students] == list(seeds):
        if len({row["data_sha256"] for row in students}) != 1 or len({row["teacher_sha256"] for row in students}) != 1:
            raise ValueError("Cannot summarize students with different data or teacher hashes")
        isolated = students[0]
        if isolated["parallel_jobs"] != 1:
            raise ValueError("Seed 0 must run alone for the cost reference")
        summary.append({"dataset": dataset, "method": "maskedkd", "data_sha256": isolated["data_sha256"],
                        "teacher_sha256": isolated["teacher_sha256"],
                        "seeds": "|".join(str(row["seed"]) for row in students),
                        "test_primary_mean": statistics.mean(row["test_primary"] for row in students),
                        "test_primary_sd": statistics.stdev(row["test_primary"] for row in students),
                        "test_overall_mean": statistics.mean(row["test_overall"] for row in students),
                        "test_overall_sd": statistics.stdev(row["test_overall"] for row in students),
                        "isolated_seed0_train_hours": isolated["train_hours"],
                        "isolated_seed0_peak_vram_gib": isolated["peak_vram_gib"],
                        "isolated_seed0_samples_per_second": isolated["samples_per_second"]})
    write_csv(destination / f"{dataset}_summary.csv", SUMMARY_FIELDS, summary)
    representative = run_directory(output_root, dataset, "student", 0) / "best.pt"
    if publish_checkpoint and representative.exists():
        destination = ROOT / "checkpoints" / dataset / "maskedkd_seed0.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != sha256(representative):
            raise ValueError(f"Existing published checkpoint differs: {destination}")
        if not destination.exists():
            shutil.copyfile(representative, destination)
    print(f"Exported {len(main)} completed runs for {dataset} into results/", flush=True)
