#!/usr/bin/env python3
"""Download and validate the original Waterbirds release from group_DRO."""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_assets import check_waterbirds


URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
NAME = "waterbird_complete95_forest2water2"


def safe_extract(archive, destination):
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive entry: {member.name}")
            if relative in seen and member.isfile():
                raise ValueError(f"Duplicate archive entry: {member.name}")
            seen.add(relative)
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as input_file, target.open("wb") as output_file:
                    shutil.copyfileobj(input_file, output_file)


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        check_waterbirds(output, None)
        print(f"REUSED verified dataset: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = output.parent / f"{NAME}.tar.gz"
    part = archive.with_suffix(archive.suffix + ".part")
    if not archive.exists():
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
                        "--output", str(part), URL], check=True)
        with tarfile.open(part, "r:gz") as source:
            for member in source:
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unexpected archive entry: {member.name}")
        os.replace(part, archive)
    with tempfile.TemporaryDirectory(prefix="waterbirds-", dir=output.parent) as staging_name:
        staging = Path(staging_name)
        safe_extract(archive, staging)
        roots = [path.parent for path in staging.rglob("metadata.csv")]
        if len(roots) != 1:
            raise ValueError("Expected exactly one Waterbirds metadata.csv in the archive")
        check_waterbirds(roots[0], None)
        os.replace(roots[0], output)
    print(f"Prepared Waterbirds: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "data" / NAME)
    args = parser.parse_args()
    prepare(args.output)
