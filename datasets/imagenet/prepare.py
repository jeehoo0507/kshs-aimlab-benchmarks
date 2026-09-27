"""Prepare only the labeled ImageNet-1K 2012 train and validation splits."""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from check_assets import check_imagenet  # noqa: E402


ARCHIVES = {
    "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf"),
}
IMAGE = re.compile(r"ILSVRC2012_(val|test)_(\d{8})\.JPEG\Z")
TRAIN_IMAGE = re.compile(r"n\d{8}_\d+\.JPEG\Z")
WNID = re.compile(r"n\d{8}\Z")


def archive_hashes(path):
    md5, sha256 = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def read_devkit(path):
    prefix = "ILSVRC2012_devkit_t12/data/"
    with tarfile.open(path, "r:gz") as archive:
        meta = archive.extractfile(prefix + "meta.mat")
        labels = archive.extractfile(prefix + "ILSVRC2012_validation_ground_truth.txt")
        if meta is None or labels is None:
            raise ValueError("Devkit is missing class metadata or validation labels")
        synsets = loadmat(io.BytesIO(meta.read()), squeeze_me=True)["synsets"]
        ground_truth = [int(line) for line in labels.read().splitlines()]
    id_to_wnid = {int(row[0]): str(row[1]) for row in synsets if int(row[4]) == 0}
    if len(id_to_wnid) != 1000 or len(set(id_to_wnid.values())) != 1000:
        raise ValueError("Devkit must define 1,000 unique leaf classes")
    if not all(WNID.fullmatch(wnid) for wnid in id_to_wnid.values()):
        raise ValueError("Invalid WordNet ID in devkit")
    if len(ground_truth) != 50000 or any(label not in id_to_wnid for label in ground_truth):
        raise ValueError("Devkit must label all 50,000 validation images")
    return id_to_wnid, ground_truth


def copy_member(archive, member, destination):
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Cannot read image from archive: {member.name}")
    with source, destination.open("wb") as output:
        shutil.copyfileobj(source, output, 4 * 1024 * 1024)


def prepare_val(path, root, id_to_wnid, labels):
    expected_total = len(labels)
    expected_per_class, remainder = divmod(expected_total, len(id_to_wnid))
    if remainder or expected_per_class == 0:
        raise ValueError("Validation labels are not balanced across classes")
    final = root / "val"
    if final.exists():
        return
    stage = root / ".val.partial"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        counts, seen = Counter(), set()
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                if not member.isfile() or Path(member.name).name != member.name:
                    raise ValueError(f"Unexpected validation archive member: {member.name}")
                match = IMAGE.fullmatch(member.name)
                if match is None or match.group(1) != "val":
                    raise ValueError(f"Unexpected validation image: {member.name}")
                number = int(match.group(2))
                if not 1 <= number <= expected_total or number in seen:
                    raise ValueError(f"Invalid or duplicate validation image: {member.name}")
                seen.add(number)
                wnid = id_to_wnid[labels[number - 1]]
                class_dir = stage / wnid
                class_dir.mkdir(exist_ok=True)
                copy_member(archive, member, class_dir / member.name)
                counts[wnid] += 1
        if (len(seen) != expected_total or set(counts) != set(id_to_wnid.values())
                or set(counts.values()) != {expected_per_class}):
            raise ValueError("Validation archive does not match the devkit labels")
        os.replace(stage, final)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def prepare_train(path, root, wnids):
    train = root / "train"
    train.mkdir(exist_ok=True)
    seen = set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).name != member.name or not member.name.endswith(".tar"):
                raise ValueError(f"Unexpected training archive member: {member.name}")
            wnid = member.name.removesuffix(".tar")
            if wnid not in wnids or wnid in seen:
                raise ValueError(f"Unknown or duplicate training class: {wnid}")
            seen.add(wnid)
            final = train / wnid
            if final.exists():
                continue  # A completed class from an earlier run.
            stage = train / f".{wnid}.partial"
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir()
            try:
                nested = archive.extractfile(member)
                if nested is None:
                    raise ValueError(f"Cannot read nested archive: {member.name}")
                count = 0
                with nested, tarfile.open(fileobj=nested, mode="r|*") as images:
                    for image in images:
                        if (not image.isfile() or Path(image.name).name != image.name
                                or not TRAIN_IMAGE.fullmatch(image.name)
                                or not image.name.startswith(wnid + "_")):
                            raise ValueError(f"Unexpected training image: {image.name}")
                        destination = stage / image.name
                        if destination.exists():
                            raise ValueError(f"Duplicate training image: {image.name}")
                        copy_member(images, image, destination)
                        count += 1
                if count == 0:
                    raise ValueError(f"Empty training class: {wnid}")
                os.replace(stage, final)
            except Exception:
                shutil.rmtree(stage, ignore_errors=True)
                raise
            if len(seen) % 50 == 0:
                print(f"Prepared {len(seen)}/1000 training classes", flush=True)
    if seen != wnids:
        raise ValueError("Training archive does not contain all 1,000 classes")


def prepare(source, root):
    source, root = source.resolve(), root.resolve()
    if (root / "manifest.json").exists():
        check_imagenet(root)
        print(f"Reused verified ImageNet-1K train/val: {root}")
        return
    paths = {name: source / filename for name, (filename, _) in ARCHIVES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Download the official ImageNet-1K train, val and devkit archives first:\n" + "\n".join(missing))
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, path in paths.items():
        md5, sha256 = archive_hashes(path)
        if md5 != ARCHIVES[name][1]:
            raise ValueError(f"Official MD5 mismatch: {path}")
        hashes[name] = sha256
        print(f"Verified {path.name}", flush=True)
    id_to_wnid, labels = read_devkit(paths["devkit"])
    prepare_val(paths["val"], root, id_to_wnid, labels)
    prepare_train(paths["train"], root, set(id_to_wnid.values()))
    manifest = {
        "schema": 1,
        "dataset": "imagenet2012_train_val",
        "classes": sorted(id_to_wnid.values()),
        "counts": {"train": 1281167, "val": 50000},
        "archive_sha256": hashes,
    }
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, root / "manifest.json")
    try:
        check_imagenet(root)
    except Exception:
        (root / "manifest.json").unlink()
        raise
    print(f"Prepared ImageNet-1K train/val: {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="directory with three official archives")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "data/imagenet")
    args = parser.parse_args()
    prepare(args.source, args.output)
