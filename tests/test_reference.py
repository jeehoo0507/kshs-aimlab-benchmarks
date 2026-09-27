import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from PIL import Image

from reference.data import BenchmarkDataset
from reference.engine import train_run
from reference.report import export


def make_coco(root):
    rows = []
    classes = [f"class_{i}" for i in range(10)]
    for label in range(10):
        for split in ("train", "val", "test"):
            stem = f"{split}_{label}"
            image = root / "images" / f"{stem}.png"
            mask = root / "masks" / f"{stem}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), (label * 15, 60, 90)).save(image)
            Image.new("L", (32, 32), 255).save(mask)
            rows.append({"id": len(rows), "split": split, "label": label,
                         "probe": split == "val", "image": str(image.relative_to(root)),
                         "mask": str(mask.relative_to(root))})
        extra_image = root / "images" / f"val_extra_{label}.png"
        extra_mask = root / "masks" / f"val_extra_{label}.png"
        Image.new("RGB", (32, 32), (label * 15, 60, 90)).save(extra_image)
        Image.new("L", (32, 32), 255).save(extra_mask)
        rows.append({"id": len(rows), "split": "val", "label": label, "probe": False,
                     "image": str(extra_image.relative_to(root)),
                     "mask": str(extra_mask.relative_to(root))})
    (root / "manifest.json").write_text(json.dumps({"selection": "one_annotated_instance_per_image",
                                                    "classes": classes, "images": rows}))


def test_cpu_reference_train_resume_and_csv(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    make_coco(data)
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/reference.json").read_text())
    cfg.update(teacher_epochs=1, student_epochs=1, batch_size=2, eval_batch_size=5,
               accumulation_steps=2, warmup_epochs=0, probe_epochs=[0, 1])
    output = tmp_path / "outputs"
    teacher = train_run("coco", data, output, cfg, "teacher", 0, "cpu", debug=True)
    student = train_run("coco", data, output, cfg, "student", 0, "cpu", debug=True)
    assert teacher["best_epoch"] == student["best_epoch"] == 1
    assert len(student["best_test"]["class_counts"]) == 10
    assert student["test_mask"]["n"] == 10
    assert student["test_mask"]["foreground_recall"] == 0.5
    test_mask = json.loads((output / "coco/maskedkd/seed_0/test_mask.json").read_text())
    assert len(test_mask["sample_ids"]) == len(test_mask["selection_hex"]) == 10
    assert all(len(values) == 10 for values in test_mask["predictions"].values())
    assert len(json.loads((output / "coco/maskedkd/seed_0/probes.json").read_text())[0]["selection_hex"]) == 20
    assert train_run("coco", data, output, cfg, "student", 0, "cpu", debug=True) == student
    for seed in (1, 2):
        train_run("coco", data, output, cfg, "student", seed, "cpu", debug=True)
    import reference.report as report
    monkeypatch.setattr(report, "ROOT", tmp_path)
    export(output, "coco", seeds=(0, 1, 2), publish_checkpoint=True)
    with (tmp_path / "results/coco/runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["method"] for row in rows} == {"teacher", "maskedkd"}
    assert len(rows) == 4
    with (tmp_path / "results/coco/summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 1 and summary[0]["seeds"] == "0|1|2"
    assert (tmp_path / "checkpoints/coco/maskedkd_coco.pt").is_file()
    with (tmp_path / "results/coco/mask_validation.csv").open() as stream:
        assert {int(row["epoch"]) for row in csv.DictReader(stream)} == {0, 1}
    with (tmp_path / "results/coco/mask_test.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3


def test_waterbirds_full_validation_and_test_with_masks(tmp_path):
    rows = []
    masks = tmp_path / "segmentations"
    masks.mkdir()
    for group in range(4):
        for split in (0, 1, 2):
            count = 20 if split == 1 else 1
            for index in range(count):
                name = f"g{group}_s{split}_{index}.png"
                Image.new("RGB", (32, 32), (group * 40, 50, 90)).save(tmp_path / name)
                Image.new("L", (32, 32), 255).save(masks / name)
                rows.append({"img_filename": name, "split": split,
                             "y": group // 2, "place": group % 2})
    with (tmp_path / "metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("img_filename", "split", "y", "place"))
        writer.writeheader()
        writer.writerows(rows)
    validation = BenchmarkDataset("waterbirds", tmp_path, "val", with_foreground=True, seg_root=masks)
    assert len(validation) == 80
    item = validation[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["foreground"].shape == (196,)
    assert item["foreground"].all()
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/reference.json").read_text())
    cfg.update(teacher_epochs=1, student_epochs=1, batch_size=2, eval_batch_size=16,
               accumulation_steps=2, warmup_epochs=0, probe_epochs=[0, 1])
    output = tmp_path / "outputs"
    train_run("waterbirds", tmp_path, output, cfg, "teacher", 0, "cpu", debug=True)
    result = train_run("waterbirds", tmp_path, output, cfg, "student", 0, "cpu", debug=True,
                       seg_root=masks)
    assert len(result["best_test"]["group_accuracy"]) == 4
    assert result["best_test"]["worst_group_accuracy"] == min(result["best_test"]["group_accuracy"])
    assert result["test_mask"]["n"] == 4
    assert result["test_mask"]["foreground_recall"] == 0.5
    assert len(json.loads((output / "waterbirds/maskedkd/seed_0/probes.json").read_text())[0]["selection_hex"]) == 80


def test_interrupted_epoch_restores_selected_checkpoint(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    make_coco(data)
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/reference.json").read_text())
    cfg.update(teacher_epochs=1, student_epochs=2, batch_size=2, eval_batch_size=5,
               accumulation_steps=2, warmup_epochs=0, probe_epochs=[0, 1, 2])
    output = tmp_path / "outputs"
    train_run("coco", data, output, cfg, "teacher", 0, "cpu", debug=True)
    import reference.engine as engine
    original = engine.train_epoch

    def interrupt(model, teacher, dataset, optimizer, scaler, config, device, epoch):
        if teacher is not None and epoch == 2:
            raise RuntimeError("simulated interruption")
        return original(model, teacher, dataset, optimizer, scaler, config, device, epoch)

    monkeypatch.setattr(engine, "train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_run("coco", data, output, cfg, "student", 0, "cpu", debug=True)
    directory = output / "coco/maskedkd/seed_0"
    assert engine.load_checkpoint(directory / "resume.pt")["epoch"] == 1
    # Simulate stale writes beyond the durable resume state.
    engine.save_checkpoint(directory / "best.pt", {"epoch": 99, "model": {}})
    (directory / "history.json").write_text("[]")
    monkeypatch.setattr(engine, "train_epoch", original)
    result = train_run("coco", data, output, cfg, "student", 0, "cpu", debug=True)
    assert len(json.loads((directory / "history.json").read_text())) == 2
    assert engine.load_checkpoint(directory / "best.pt")["epoch"] == result["best_epoch"]


def test_auto_jobs_respects_measured_vram(monkeypatch):
    import reference.cli as cli
    monkeypatch.setattr(cli, "benchmark_memory", lambda args: 5.0)
    monkeypatch.setattr(cli, "free_gib", lambda: 18.0)
    config = Path(__file__).resolve().parents[1] / "configs/reference.json"
    assert cli.choose_jobs(SimpleNamespace(device="cuda", jobs="auto", config=config))[0] == 2
    with pytest.raises(ValueError, match="safe maximum=2"):
        cli.choose_jobs(SimpleNamespace(device="cuda", jobs="3", config=config))
    monkeypatch.setattr(cli, "benchmark_memory", lambda args, role="student": 5.0)
    cli.check_teacher_memory(SimpleNamespace())
    monkeypatch.setattr(cli, "free_gib", lambda: 8.0)
    with pytest.raises(RuntimeError, match="Teacher needs at least"):
        cli.check_teacher_memory(SimpleNamespace())
