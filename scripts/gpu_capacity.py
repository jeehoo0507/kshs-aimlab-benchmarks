"""Calculate a GPU job limit from a measured one-job VRAM peak."""

import argparse
import math
import subprocess


GLOBAL_RESERVE_GIB = 2.5
PEAK_MULTIPLIER = 1.25
PER_JOB_OVERHEAD_GIB = 0.5


def job_budget_gib(peak_gib):
    if not math.isfinite(peak_gib) or peak_gib <= 0:
        raise ValueError("One-job peak VRAM must be positive and finite")
    return peak_gib * PEAK_MULTIPLIER + PER_JOB_OVERHEAD_GIB


def memory_job_limit(free_gib, peak_gib):
    if not math.isfinite(free_gib) or free_gib < 0:
        raise ValueError("Free VRAM must be nonnegative and finite")
    return max(0, math.floor((free_gib - GLOBAL_RESERVE_GIB) / job_budget_gib(peak_gib)))


def free_gib_from_nvidia_smi(gpu):
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip()) / 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-gib", type=float, required=True,
                        help="peak reserved VRAM (GiB) measured for one representative job")
    parser.add_argument("--pending", type=int, help="number of independent jobs waiting to run")
    parser.add_argument("--gpu", default="0", help="nvidia-smi GPU index or UUID (default: 0)")
    parser.add_argument("--free-gib", type=float,
                        help="override current free VRAM, mainly for offline planning")
    args = parser.parse_args()
    if args.pending is not None and args.pending < 0:
        parser.error("--pending must be nonnegative")
    try:
        free = args.free_gib if args.free_gib is not None else free_gib_from_nvidia_smi(args.gpu)
        budget = job_budget_gib(args.peak_gib)
        capacity = memory_job_limit(free, args.peak_gib)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"GPU capacity check failed: {exc}\n")
    launch = min(capacity, args.pending) if args.pending is not None else capacity
    print(f"Free={free:.2f} GiB; one-job peak={args.peak_gib:.2f} GiB; "
          f"budget/job={budget:.2f} GiB; reserve={GLOBAL_RESERVE_GIB:.1f} GiB")
    print(f"Memory-limited maximum={capacity}; jobs to launch={launch}")
    if args.pending is not None:
        print(f"Pending jobs={args.pending}")


if __name__ == "__main__":
    main()
