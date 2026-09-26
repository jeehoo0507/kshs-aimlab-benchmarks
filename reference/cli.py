"""One-command baseline orchestration for a single CUDA GPU."""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import torch
from torch.nn import functional as F

from .engine import ROOT, load_checkpoint, run_directory, train_run
from .model import URLS, build_model
from .report import export


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg["keep_patches"] != 98 or cfg["seeds"] != [0, 1, 2]:
        raise ValueError("Reference runs require 98 kept patches and seeds 0, 1, 2")
    return cfg


def data_root_for(dataset, supplied):
    return Path(supplied).expanduser().resolve() if supplied else ROOT / "data" / (
        "coco_single" if dataset == "coco" else "waterbird_complete95_forest2water2")


def prepare_data(dataset, data_root, supplied):
    marker = data_root / ("manifest.json" if dataset == "coco" else "metadata.csv")
    if not marker.is_file():
        if supplied:
            raise FileNotFoundError(f"Dataset missing at --data-root {data_root}")
        script = ROOT / "datasets" / ("coco_single" if dataset == "coco" else "waterbirds") / "setup.sh"
        print(f"Preparing {dataset} dataset with {script.relative_to(ROOT)}", flush=True)
        subprocess.run(["bash", str(script)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_assets.py"), dataset, str(data_root)],
                   cwd=ROOT, check=True)


def prefetch():
    for role in ("teacher", "student"):
        print(f"Caching official DeiT {role} ImageNet weights", flush=True)
        torch.hub.load_state_dict_from_url(URLS[role], map_location="cpu", check_hash=True, weights_only=True)


def worker_command(args, role, seed, debug=False):
    return [sys.executable, "-m", "reference.cli", "_worker", "--dataset", args.dataset,
            "--data-root", str(args.data_root), "--output-root", str(args.output_root),
            "--config", str(args.config), "--device", args.device,
            "--role", role, "--seed", str(seed)] + (["--debug"] if debug else [])


def failed_log(path):
    try:
        lines = deque(Path(path).read_text().splitlines(), maxlen=35)
        return "\n".join(lines)
    except OSError:
        return f"Log unavailable: {path}"


def run_visible(command, log):
    with log.open("a") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        return process.wait()


def run_teacher(args):
    directory = run_directory(args.output_root, args.dataset, "teacher", 0)
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "train.log"
    print(f"Teacher DeiT-S, 30 epochs → {log}", flush=True)
    code = run_visible(worker_command(args, "teacher", 0), log)
    if code:
        raise RuntimeError(f"Teacher training failed ({code}):\n{failed_log(log)}")
    export(args.output_root, args.dataset, publish_checkpoint=False)


def run_seed_zero(args):
    directory = run_directory(args.output_root, args.dataset, "student", 0)
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "train.log"
    print(f"MaskedKD seed=0, isolated cost reference → {log}", flush=True)
    code = run_visible(worker_command(args, "student", 0), log)
    if code:
        raise RuntimeError(f"MaskedKD seed=0 failed ({code}):\n{failed_log(log)}")
    export(args.output_root, args.dataset)


def free_gib():
    # Called only on a single-GPU host or when CUDA_VISIBLE_DEVICES identifies GPU 0.
    free_bytes, _ = torch.cuda.mem_get_info(0)
    return free_bytes / 1024**3


def benchmark_memory(args, role="student"):
    command = [sys.executable, "-m", "reference.cli", "_benchmark", "--dataset", args.dataset,
               "--output-root", str(args.output_root), "--config", str(args.config), "--role", role]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"VRAM benchmark failed:\n{result.stderr}\n{result.stdout}")
    return json.loads(result.stdout.strip().splitlines()[-1])["peak_reserved_gib"]


def check_teacher_memory(args):
    peak = benchmark_memory(args, "teacher")
    available = free_gib()
    required = peak * 1.25 + 3.0
    if available < required:
        raise RuntimeError(f"Teacher needs at least {required:.1f} GiB free by the measured "
                           f"VRAM budget; only {available:.1f} GiB is free")
    print(f"Teacher VRAM benchmark peak={peak:.2f} GiB; free={available:.2f} GiB", flush=True)


def choose_jobs(args):
    if args.device != "cuda":
        return 1, 0
    peak = benchmark_memory(args)
    # The margin includes allocator variability, CUDA contexts, and other jobs.
    reserve_per_job = peak * 1.25 + 0.5
    available = free_gib()
    safe = max(0, min(2, int((available - 2.5) // reserve_per_job)))
    if safe < 1:
        raise RuntimeError(f"Insufficient free GPU memory: {available:.1f} GiB free; "
                           f"estimated {reserve_per_job:.1f} GiB/job plus 2.5 GiB reserve")
    requested = safe if args.jobs == "auto" else int(args.jobs)
    if requested < 1 or requested > safe:
        raise ValueError(f"Requested jobs={requested}, measured safe maximum={safe}; "
                         "do not override the GPU memory reserve")
    print(f"VRAM benchmark peak={peak:.2f} GiB; free={available:.2f} GiB; "
          f"safe jobs={safe}; launching {requested}", flush=True)
    return requested, reserve_per_job


def run_students(args, jobs, reserve_per_job):
    seeds = load_config(args.config)["seeds"][1:]
    pending = deque(seeds)
    active = {}
    last_status = time.monotonic()
    try:
        while pending or active:
            while pending and len(active) < jobs:
                if args.device == "cuda" and free_gib() < reserve_per_job + 2.5:
                    if active:
                        break
                    raise RuntimeError("Free GPU memory fell below the per-job reserve; retry later")
                seed = pending.popleft()
                directory = run_directory(args.output_root, args.dataset, "student", seed)
                directory.mkdir(parents=True, exist_ok=True)
                log = directory / "train.log"
                stream = log.open("a")
                process = subprocess.Popen(worker_command(args, "student", seed), cwd=ROOT,
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           env={**os.environ, "REFERENCE_PARALLEL_JOBS": str(jobs)})
                active[seed] = (process, stream, log)
                print(f"START MaskedKD seed={seed} pid={process.pid}; log={log}", flush=True)
                # Let CUDA allocations appear before deciding whether to start another job.
                time.sleep(3)
            finished = []
            for seed, (process, stream, log) in active.items():
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                finished.append(seed)
                if code:
                    raise RuntimeError(f"MaskedKD seed={seed} failed ({code}):\n{failed_log(log)}")
                print(f"DONE MaskedKD seed={seed}", flush=True)
                export(args.output_root, args.dataset)
            for seed in finished:
                del active[seed]
            if active and not finished:
                if time.monotonic() - last_status >= 60:
                    for seed in active:
                        history = run_directory(args.output_root, args.dataset, "student", seed) / "history.json"
                        epoch = len(json.loads(history.read_text())) if history.exists() else 0
                        print(f"RUNNING MaskedKD seed={seed} epoch={epoch}/{load_config(args.config)['student_epochs']}", flush=True)
                    last_status = time.monotonic()
                time.sleep(10)
    except BaseException:
        for process, stream, _ in active.values():
            if process.poll() is None:
                process.terminate()
            process.wait()
            stream.close()
        raise


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    cfg = load_config(args.config)
    classes = 10 if args.dataset == "coco" else 2
    device = torch.device("cuda")
    if args.role == "teacher":
        model = build_model("teacher", classes, pretrained=True, drop_path=cfg["drop_path"]).to(device)
        teacher = None
    else:
        teacher_file = run_directory(args.output_root, args.dataset, "teacher", 0) / "best.pt"
        teacher = build_model("teacher", classes, drop_path=cfg["drop_path"]).to(device).eval().requires_grad_(False)
        teacher.load_state_dict(load_checkpoint(teacher_file)["model"])
        model = build_model("student", classes, pretrained=True, drop_path=cfg["drop_path"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scaler = torch.amp.GradScaler("cuda")
    images = torch.randn(cfg["batch_size"], 3, 224, 224, device=device)
    labels = torch.arange(cfg["batch_size"], device=device) % classes
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.float16):
            if teacher is None:
                logits = model(images)
                loss = F.cross_entropy(logits.float(), labels)
            else:
                logits, attention = model(images, return_attention=True)
                with torch.no_grad():
                    targets = teacher(images, attention.detach().topk(cfg["keep_patches"], 1).indices)
                loss = 0.5 * F.cross_entropy(logits.float(), labels) + 0.5 * F.kl_div(
                    F.log_softmax(logits.float(), 1), F.softmax(targets.float(), 1), reduction="batchmean")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(json.dumps({"peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "export", "status", "_worker", "_benchmark"):
        q = sub.add_parser(name)
        q.add_argument("--dataset", choices=("coco", "waterbirds"), required=True)
        q.add_argument("--data-root")
        q.add_argument("--output-root", default="outputs/reference")
        q.add_argument("--config", default="configs/reference.json")
        if name == "run":
            q.add_argument("--jobs", default="auto")
            q.add_argument("--device", choices=("cuda",), default="cuda")
        if name == "_worker":
            q.add_argument("--role", choices=("teacher", "student"), required=True)
            q.add_argument("--seed", type=int, required=True)
            q.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
            q.add_argument("--debug", action="store_true")
        if name == "_benchmark":
            q.add_argument("--role", choices=("teacher", "student"), default="student")
    return p


def main():
    args = parser().parse_args()
    args.output_root = Path(args.output_root).expanduser().resolve()
    args.config = Path(args.config).expanduser().resolve()
    supplied_data_root = args.data_root is not None
    args.data_root = data_root_for(args.dataset, args.data_root)
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
    cfg = load_config(args.config)
    if args.command == "_worker":
        result = train_run(args.dataset, args.data_root, args.output_root, cfg,
                           args.role, args.seed, args.device, args.debug)
        print(json.dumps({"role": args.role, "seed": args.seed,
                          "best_epoch": result["best_epoch"], "best_validation": result["best_validation"]}))
    elif args.command == "_benchmark":
        benchmark(args)
    elif args.command == "export":
        export(args.output_root, args.dataset, cfg["seeds"])
    elif args.command == "status":
        for role, seed in [("teacher", 0), *(("student", s) for s in cfg["seeds"])]:
            directory = run_directory(args.output_root, args.dataset, role, seed)
            history = directory / "history.json"
            completed = (directory / "result.json").exists()
            epoch = len(json.loads(history.read_text())) if history.exists() else 0
            total = cfg["teacher_epochs"] if role == "teacher" else cfg["student_epochs"]
            print(f"{role:7} seed={seed}: {epoch}/{total} {'complete' if completed else 'pending/running'}")
    elif args.command == "run":
        prepare_data(args.dataset, args.data_root, supplied_data_root)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; install the CUDA build of PyTorch")
        prefetch()
        check_teacher_memory(args)
        run_teacher(args)
        jobs, reserve = choose_jobs(args)
        if free_gib() < reserve + 2.5:
            raise RuntimeError("Free GPU memory fell below the isolated-student reserve; retry later")
        run_seed_zero(args)
        run_students(args, jobs, reserve)
        export(args.output_root, args.dataset)


if __name__ == "__main__":
    main()
