"""Resumable, validation-selected MaskedKD reference runs."""
import hashlib
import csv
import json
import math
import os
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import BenchmarkDataset, loader
from .metrics import autocast, evaluate, mask_probe, selection_overlap
from .model import URLS, build_model


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def code_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "reference").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_commit():
    import subprocess
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def metadata_hash(name, root):
    return sha256(Path(root) / ("manifest.json" if name == "coco" else "metadata.csv"))


def foreground_hash(name, root, seg_root):
    if name == "coco":
        rows = json.loads((Path(root) / "manifest.json").read_text())["images"]
        paths = [(row["mask"], Path(root) / row["mask"]) for row in rows
                 if row["split"] in ("val", "test")]
    else:
        if seg_root is None:
            raise ValueError("Waterbirds foreground diagnostics require --seg-root")
        with (Path(root) / "metadata.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        paths = [(str(Path(row["img_filename"]).with_suffix(".png")),
                  Path(seg_root) / Path(row["img_filename"]).with_suffix(".png"))
                 for row in rows if int(row["split"]) in (1, 2)]
    digest = hashlib.sha256()
    for relative, path in paths:
        digest.update(relative.encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def pretrain_hash(role, debug=False):
    if debug:
        return "random-debug"
    path = Path(torch.hub.get_dir()) / "checkpoints" / URLS[role].rsplit("/", 1)[-1]
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained weights missing: {path}")
    return sha256(path)


def selection_metric(name):
    return "macro_accuracy" if name == "coco" else "worst_group_accuracy"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate(cfg, epoch, total):
    base = cfg["learning_rate"]
    warmup = min(cfg["warmup_epochs"], max(total - 1, 0))
    if epoch <= warmup:
        return base * epoch / max(1, warmup)
    fraction = (epoch - warmup - 1) / max(1, total - warmup - 1)
    floor = min(base, cfg["min_learning_rate"])
    return floor + (base - floor) * (1 + math.cos(math.pi * fraction)) / 2


def optimizer_for(model, cfg):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim == 1 or name.endswith(".bias") or name in ("pos_embed", "cls_token") else decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}], lr=cfg["learning_rate"])


def train_epoch(model, teacher, dataset, optimizer, scaler, cfg, device, epoch):
    model.train()
    seed_all(cfg["seed"] + epoch * 100003)
    optimizer.zero_grad(set_to_none=True)
    batches = loader(dataset, cfg["batch_size"], training=True)
    total = len(batches)
    seen = correct = updates = 0
    sums = {"loss": 0.0, "ce": 0.0, "kd": 0.0}
    started = time.monotonic()
    for step, batch in enumerate(batches):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with autocast(device):
            if teacher is None:
                logits = model(images)
                teacher_logits = None
            else:
                logits, attention = model(images, return_attention=True)
                indices = attention.detach().topk(cfg["keep_patches"], dim=1).indices
                with torch.no_grad():
                    teacher_logits = teacher(images, indices)
            ce = F.cross_entropy(logits.float(), labels, label_smoothing=cfg["label_smoothing"])
            kd = logits.new_zeros(())
            if teacher_logits is not None:
                temperature = cfg["temperature"]
                kd = F.kl_div(F.log_softmax(logits.float() / temperature, 1),
                              F.softmax(teacher_logits.float() / temperature, 1),
                              reduction="batchmean") * temperature**2
            loss = ce if teacher_logits is None else (1 - cfg["kd_alpha"]) * ce + cfg["kd_alpha"] * kd
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {step}")
        start = step // cfg["accumulation_steps"] * cfg["accumulation_steps"]
        end = min(start + cfg["accumulation_steps"], total)
        window_samples = min(end * cfg["batch_size"], len(dataset)) - start * cfg["batch_size"]
        scaler.scale(loss * len(images) / window_samples).backward()
        if step + 1 == end:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        for key, value in (("loss", loss), ("ce", ce), ("kd", kd)):
            sums[key] += value.detach().item() * len(images)
        correct += logits.argmax(1).eq(labels).sum().item()
        seen += len(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.monotonic() - started
    return {**{key: value / seen for key, value in sums.items()}, "accuracy": correct / seen,
            "samples": seen, "optimizer_updates": updates, "seconds": seconds,
            "samples_per_second": seen / seconds}


def run_directory(output_root, dataset, role, seed):
    base = Path(output_root) / dataset
    return base / "teacher" if role == "teacher" else base / "maskedkd" / f"seed_{seed}"


def train_run(dataset, data_root, output_root, cfg, role, seed, device_name="cuda", debug=False,
              seg_root=None):
    if role not in ("teacher", "student"):
        raise ValueError(role)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a CUDA PyTorch build or use --device cpu for smoke tests")
    torch.set_num_threads(2)
    device = torch.device(device_name)
    directory = run_directory(output_root, dataset, role, seed)
    directory.mkdir(parents=True, exist_ok=True)
    data_sha = metadata_hash(dataset, data_root)
    foreground_sha = foreground_hash(dataset, data_root, seg_root) if role == "student" else ""
    teacher_file = run_directory(output_root, dataset, "teacher", 0) / "best.pt"
    if role == "student" and not teacher_file.is_file():
        raise FileNotFoundError(f"Frozen teacher missing: {teacher_file}")
    teacher_sha = sha256(teacher_file) if role == "student" else ""
    role_cfg = {**cfg, "seed": seed, "dataset": dataset, "role": role, "debug": debug}
    fingerprint = hashlib.sha256(json.dumps({"config": role_cfg, "data": data_sha,
                                             "teacher": teacher_sha, "foreground": foreground_sha,
                                             "code": code_hash()},
                                            sort_keys=True).encode()).hexdigest()
    result_path = directory / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text())
        if existing["signature"] != fingerprint:
            raise ValueError(f"Completed run provenance changed: {directory}")
        print(f"REUSE {role} {dataset} seed={seed}", flush=True)
        return existing
    if shutil.disk_usage(directory).free < 2 * 1024**3 and not debug:
        raise OSError("Less than 2 GiB free at the output path")
    resume_path = directory / "resume.pt"
    resume = load_checkpoint(resume_path) if resume_path.is_file() else None
    if resume and resume["signature"] != fingerprint:
        raise ValueError(f"Resume provenance changed: {directory}")
    seed_all(seed)
    classes = 10 if dataset == "coco" else 2
    model = build_model(role, classes, pretrained=resume is None and not debug,
                        drop_path=cfg["drop_path"], debug=debug).to(device)
    initial_sha = pretrain_hash(role, debug)
    teacher = None
    if role == "student":
        teacher = build_model("teacher", classes, drop_path=cfg["drop_path"], debug=debug).to(device)
        teacher.load_state_dict(load_checkpoint(teacher_file)["model"])
        teacher.requires_grad_(False).eval()
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, probes, best_epoch, best_score, start_epoch, best_weights = [], [], 0, -1.0, 0, None
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        history, probes = resume["history"], resume["probes"]
        best_epoch, best_score, start_epoch = resume["best_epoch"], resume["best_score"], resume["epoch"]
        best_weights = resume["best_model"]
        # Reconcile writes that may have reached best.pt/history.json after the last
        # durable resume checkpoint when a process was interrupted mid-epoch.
        save_checkpoint(directory / "best.pt", {"model": best_weights, "epoch": best_epoch,
                                                "signature": fingerprint, "dataset_sha256": data_sha,
                                                "teacher_sha256": teacher_sha})
        write_json(directory / "history.json", history)
        write_json(directory / "probes.json", probes)
        print(f"RESUME {role} {dataset} seed={seed} after epoch {start_epoch}", flush=True)
    train_data = BenchmarkDataset(dataset, data_root, "train", training=True)
    val_data = BenchmarkDataset(dataset, data_root, "val")
    test_data = BenchmarkDataset(dataset, data_root, "test")
    # Keep validation membership and ordering fixed across all seeds and epochs.
    probe_data = (BenchmarkDataset(dataset, data_root, "val", with_foreground=True,
                                   seg_root=seg_root) if role == "student" else None)
    if probe_data is not None:
        write_json(directory / "validation_sample_ids.json",
                   [row["image"] if dataset == "coco" else row["img_filename"]
                    for row in probe_data.records])
    total_epochs = cfg["teacher_epochs"] if role == "teacher" else cfg["student_epochs"]
    scheduled_probes = {epoch for epoch in cfg["probe_epochs"] if epoch <= total_epochs}
    if role == "student" and start_epoch == 0 and 0 in scheduled_probes and not probes:
        initial_probe = mask_probe(model, teacher, probe_data, device, cfg["eval_batch_size"], cfg["keep_patches"])
        probes.append({"epoch": 0, **initial_probe,
                       "selection_overlap_initial": 1.0, "selection_overlap_previous": 1.0})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch + 1, total_epochs + 1):
        begun = time.monotonic()
        lr = learning_rate(cfg, epoch, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        training = train_epoch(model, teacher, train_data, optimizer, scaler, {**cfg, "seed": seed}, device, epoch)
        validation = evaluate(model, val_data, device, cfg["eval_batch_size"])
        score = validation[selection_metric(dataset)]
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_weights = cpu_state(model)
            save_checkpoint(directory / "best.pt", {"model": best_weights,
                                                    "epoch": epoch, "signature": fingerprint,
                                                    "dataset_sha256": data_sha, "teacher_sha256": teacher_sha})
        if probe_data is not None and epoch in scheduled_probes:
            measured = mask_probe(model, teacher, probe_data, device,
                                  cfg["eval_batch_size"], cfg["keep_patches"])
            probes.append({"epoch": epoch, **measured,
                           "selection_overlap_initial": selection_overlap(
                               measured["selection_hex"], probes[0]["selection_hex"], cfg["keep_patches"]),
                           "selection_overlap_previous": selection_overlap(
                               measured["selection_hex"], probes[-1]["selection_hex"], cfg["keep_patches"])})
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {"epoch": epoch, "lr": lr, "train": training, "validation": validation,
               "epoch_seconds": time.monotonic() - begun}
        history.append(row)
        save_checkpoint(resume_path, {"signature": fingerprint, "epoch": epoch,
                                      "model": cpu_state(model),
                                      "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                                      "history": history, "probes": probes,
                                      "best_epoch": best_epoch, "best_score": best_score,
                                      "best_model": best_weights})
        write_json(directory / "history.json", history)
        write_json(directory / "probes.json", probes)
        print(f"{dataset} {role} seed={seed} {epoch}/{total_epochs} "
              f"val_{selection_metric(dataset)}={score:.4f} epoch={row['epoch_seconds']:.1f}s", flush=True)
    # Test is evaluated only after the validation-selected checkpoint is fixed.
    last_state = cpu_state(model)
    save_checkpoint(directory / "last.pt", {"model": last_state, "epoch": total_epochs,
                                            "signature": fingerprint})
    best_checkpoint = directory / "best.pt"
    model.load_state_dict(load_checkpoint(best_checkpoint)["model"])
    best_test = evaluate(model, test_data, device, cfg["eval_batch_size"])
    test_mask = None
    if role == "student":
        diagnostic_test = BenchmarkDataset(dataset, data_root, "test", with_foreground=True,
                                           seg_root=seg_root)
        test_mask = mask_probe(model, teacher, diagnostic_test, device,
                               cfg["eval_batch_size"], cfg["keep_patches"], record_predictions=True)
        test_mask["sample_ids"] = [row["image"] if dataset == "coco" else row["img_filename"]
                                   for row in diagnostic_test.records]
        write_json(directory / "test_mask.json", test_mask)
    model.load_state_dict(last_state)
    last_test = evaluate(model, test_data, device, cfg["eval_batch_size"])
    peak = torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
    train_seconds = sum(row["train"]["seconds"] for row in history)
    result = {"signature": fingerprint, "dataset": dataset, "role": role,
              "method": "teacher" if role == "teacher" else "maskedkd", "seed": seed,
              "student_arch": "deit_tiny_patch16_224" if role == "student" else "",
              "teacher_arch": "deit_small_patch16_224", "student_init": "imagenet" if role == "student" else "",
              "dataset_sha256": data_sha, "foreground_sha256": foreground_sha,
              "teacher_sha256": teacher_sha,
              "pretrained_sha256": initial_sha, "code_sha256": code_hash(), "code_commit": source_commit(),
              "config": role_cfg, "epochs": total_epochs, "best_epoch": best_epoch,
              "train_samples": len(train_data), "class_names": train_data.classes,
              "selection_metric": selection_metric(dataset), "best_validation": best_score,
              "best_test": best_test, "last_test": last_test,
              "test_mask": {key: value for key, value in test_mask.items()
                            if key not in ("selection_hex", "predictions", "sample_ids")}
              if test_mask is not None else None,
              "train_seconds": train_seconds, "wall_seconds": sum(row["epoch_seconds"] for row in history),
              "peak_vram_gib": peak, "checkpoint_sha256": sha256(best_checkpoint),
              "parallel_jobs": int(os.environ.get("REFERENCE_PARALLEL_JOBS", "1")),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)
                              if device.type == "cuda" else "CPU"}}
    write_json(result_path, result)
    # resume.pt is retained for provenance and possible post-run inspection.
    return result
