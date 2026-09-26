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
    assert train_run("coco", data, output, cfg, "student", 0, "cpu", debug=True) == student
    for seed in (1, 2):
        train_run("coco", data, output, cfg, "student", seed, "cpu", debug=True)
    import reference.report as report
    monkeypatch.setattr(report, "ROOT", tmp_path)
    export(output, "coco", seeds=(0, 1, 2), publish_checkpoint=True)
    with (tmp_path / "results/coco.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["method"] for row in rows} == {"teacher", "maskedkd"}
    assert len(rows) == 4
    with (tmp_path / "results/coco_summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 1 and summary[0]["seeds"] == "0|1|2"
    assert (tmp_path / "checkpoints/coco/maskedkd_seed0.pt").is_file()
    with (tmp_path / "results/coco_mask_probe.csv").open() as stream:
        assert {int(row["epoch"]) for row in csv.DictReader(stream)} == {0, 1}


def test_waterbirds_group_probe_without_masks(tmp_path):
    rows = []
    for group in range(4):
        for split in (0, 1, 2):
            count = 20 if split == 1 else 1
            for index in range(count):
                name = f"g{group}_s{split}_{index}.png"
                Image.new("RGB", (32, 32), (group * 40, 50, 90)).save(tmp_path / name)
                rows.append({"img_filename": name, "split": split,
                             "y": group // 2, "place": group % 2})
    with (tmp_path / "metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("img_filename", "split", "y", "place"))
        writer.writeheader()
        writer.writerows(rows)
    probe = BenchmarkDataset("waterbirds", tmp_path, "probe")
    assert len(probe) == 80
    item = probe[0]
    assert item["image"].shape == (3, 224, 224)
    assert "foreground" not in item
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/reference.json").read_text())
    cfg.update(teacher_epochs=1, student_epochs=1, batch_size=2, eval_batch_size=16,
               accumulation_steps=2, warmup_epochs=0, probe_epochs=[0, 1])
    output = tmp_path / "outputs"
    train_run("waterbirds", tmp_path, output, cfg, "teacher", 0, "cpu", debug=True)
    result = train_run("waterbirds", tmp_path, output, cfg, "student", 0, "cpu", debug=True)
    assert len(result["best_test"]["group_accuracy"]) == 4
    assert result["best_test"]["worst_group_accuracy"] == min(result["best_test"]["group_accuracy"])


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
    assert cli.choose_jobs(SimpleNamespace(device="cuda", jobs="auto"))[0] == 2
    with pytest.raises(ValueError, match="safe maximum=2"):
        cli.choose_jobs(SimpleNamespace(device="cuda", jobs="3"))
    monkeypatch.setattr(cli, "benchmark_memory", lambda args, role="student": 5.0)
    cli.check_teacher_memory(SimpleNamespace())
    monkeypatch.setattr(cli, "free_gib", lambda: 8.0)
    with pytest.raises(RuntimeError, match="Teacher needs at least"):
        cli.check_teacher_memory(SimpleNamespace())
