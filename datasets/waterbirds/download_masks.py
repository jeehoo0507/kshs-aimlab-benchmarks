#!/usr/bin/env python3
"""Download the official CUB segmentation masks used to compose Waterbirds."""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from check_assets import check_waterbirds
from download import safe_extract


URL = "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1"
MD5 = "4d47ba1228eae64f2fa547c47bc65255"
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    value = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def prepare(dataset, output):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.is_dir():
        check_waterbirds(dataset, output)
        print(f"REUSED verified CUB segmentation: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = output.parent / "segmentations.tgz"
    partial = archive.with_suffix(".tgz.part")
    if not archive.exists():
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
                        "--output", str(partial), URL], check=True)
        if digest(partial) != MD5:
            raise ValueError("CUB segmentation archive MD5 mismatch")
        os.replace(partial, archive)
    elif digest(archive) != MD5:
        raise ValueError("Existing CUB segmentation archive MD5 mismatch")
    with tempfile.TemporaryDirectory(prefix="cub-seg-", dir=output.parent) as staging_name:
        staging = Path(staging_name)
        safe_extract(archive, staging)
        matches = [path for path in staging.rglob("segmentations") if path.is_dir()]
        if len(matches) != 1:
            raise ValueError("Expected one segmentations directory in the CUB archive")
        check_waterbirds(dataset, matches[0])
        os.replace(matches[0], output)
    print(f"Prepared CUB segmentation: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/waterbird_complete95_forest2water2")
    parser.add_argument("--output", type=Path, default=ROOT / "data/CUB_200_2011/segmentations")
    args = parser.parse_args()
    prepare(args.dataset, args.output)
