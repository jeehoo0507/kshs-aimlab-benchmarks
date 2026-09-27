#!/usr/bin/env python3
"""Validate prepared COCO single, Waterbirds, and ImageNet datasets."""

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path


CLASSES = ("giraffe", "airplane", "clock", "zebra", "train", "bird",
           "elephant", "toilet", "stop sign", "bear")
SPLITS = {"train", "val", "test"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
WNID = re.compile(r"n\d{8}\Z")
IMAGENET_VAL = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG\Z")
IMAGENET_TRAIN = re.compile(r"n\d{8}_\d+\.JPEG\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, name):
    rel = Path(name)
    require(not rel.is_absolute() and ".." not in rel.parts and name, f"Unsafe relative path: {name}")
    root = root.resolve()
    path = (root / rel).resolve()
    require(path.is_relative_to(root), f"Path leaves root: {name}")
    require(path.is_file(), f"Missing file: {path}")
    return path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_coco(root, full):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == 1, "Unsupported COCO manifest schema")
    require(manifest.get("selection") == "one_annotated_instance_per_image",
            "COCO manifest uses an older selection rule; move the old dataset aside and rerun setup")
    require(tuple(manifest.get("classes", [])) == CLASSES,
            "COCO class set differs; move the old dataset aside and rerun setup")
    require(manifest.get("split_seed") == 20260922, "COCO split seed differs from documented definition")
    rows = manifest.get("images")
    require(isinstance(rows, list) and rows, "COCO manifest has no images")
    counts, probes, ids, decoded = Counter(), Counter(), set(), set()
    for row in rows:
        sample_id = row["id"]
        require(row.get("instances") == 1, f"COCO image does not have exactly one annotated object: {sample_id}")
        split, label = row["split"], row["label"]
        require(split in SPLITS and type(label) is int and 0 <= label < len(CLASSES),
                f"Invalid split/label: {sample_id}")
        require(sample_id not in ids, f"Duplicate COCO image id: {sample_id}")
        ids.add(sample_id)
        source = row.get("source_split")
        require(source == ("val2017" if split == "test" else "train2017"),
                f"Source split mismatch: {sample_id}")
        if "pixel_sha256" in row:
            require(row["pixel_sha256"] not in decoded, f"Duplicate decoded image: {sample_id}")
            decoded.add(row["pixel_sha256"])
        counts[split, label] += 1
        if row.get("probe"):
            require(split == "val", f"Probe outside validation: {sample_id}")
            probes[label] += 1
        for field in ("image", "mask"):
            path = inside(root, row[field])
            expected = row.get(field + "_sha256")
            require(isinstance(expected, str) and SHA256.fullmatch(expected),
                    f"Missing/invalid {field} SHA-256: {sample_id}")
            if full:
                require(digest(path) == expected, f"Changed {field}: {sample_id}")
    for split in SPLITS:
        require(all(counts[split, label] > 0 for label in range(len(CLASSES))),
                f"Missing COCO class in {split}")
    for split in ("train", "val"):
        require(len({counts[split, label] for label in range(len(CLASSES))}) == 1,
                f"Unbalanced COCO {split} split")
    require(all(probes[label] == 20 for label in range(len(CLASSES))),
            "COCO probe should contain 20 validation images per class")
    print("COCO single verified" + (" (SHA-256 checked)" if full else " (file presence)"))
    print("  " + ", ".join(f"{split}={sum(counts[split, c] for c in range(10))}" for split in ("train", "val", "test")))
    print("  train/val per class:", counts["train", 0], counts["val", 0], "; probe=20 per class")


def check_waterbirds(root, seg_root):
    with (root / "metadata.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and {"img_filename", "split", "y", "place"} <= set(reader.fieldnames),
                "Waterbirds metadata.csv lacks required columns")
        rows = list(reader)
    require(rows, "Waterbirds metadata.csv has no images")
    counts = Counter()
    seen = set()
    for row in rows:
        rel = row["img_filename"]
        require(rel not in seen, f"Duplicate Waterbirds filename: {rel}")
        seen.add(rel)
        split, y, place = int(row["split"]), int(row["y"]), int(row["place"])
        require(split in (0, 1, 2) and y in (0, 1) and place in (0, 1),
                f"Invalid Waterbirds split/group: {rel}")
        inside(root, rel)
        if seg_root is not None:
            inside(seg_root, str(Path(rel).with_suffix(".png")))
        counts[split, y, place] += 1
    for split in (0, 1, 2):
        require(all(counts[split, y, p] > 0 for y in (0, 1) for p in (0, 1)),
                f"Missing Waterbirds group in split {split}")
    print("Waterbirds verified" + (" (including segmentation)" if seg_root else " (images only)"))
    for split, name in ((0, "train"), (1, "val"), (2, "test")):
        groups = [counts[split, y, p] for y in (0, 1) for p in (0, 1)]
        print(f"  {name}: total={sum(groups)}, groups(y0p0,y0p1,y1p0,y1p1)={groups}")


def check_imagenet(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == 1 and manifest.get("dataset") == "imagenet2012_train_val",
            "Unsupported ImageNet manifest")
    classes = manifest.get("classes")
    require(isinstance(classes, list) and len(classes) == 1000
            and all(isinstance(name, str) and WNID.fullmatch(name) for name in classes)
            and classes == sorted(set(classes)),
            "ImageNet manifest needs 1,000 sorted WordNet IDs")
    require(manifest.get("counts") == {"train": 1281167, "val": 50000},
            "Unexpected ImageNet split counts")
    hashes = manifest.get("archive_sha256")
    require(isinstance(hashes, dict) and set(hashes) == {"train", "val", "devkit"}
            and all(isinstance(value, str) and SHA256.fullmatch(value) for value in hashes.values()),
            "Missing ImageNet source archive hashes")
    require(not (root / "test").exists(), "This ImageNet setup contains train and val only")

    totals, val_ids = {}, set()
    for split in ("train", "val"):
        directory = root / split
        require(directory.is_dir(), f"Missing ImageNet {split} directory")
        actual_classes = {item.name for item in os.scandir(directory) if item.is_dir()}
        require(actual_classes == set(classes), f"ImageNet {split} classes differ from manifest")
        total = 0
        for wnid in classes:
            count = 0
            with os.scandir(directory / wnid) as entries:
                for entry in entries:
                    require(entry.is_file() and not entry.is_symlink(),
                            f"Unexpected ImageNet {split} entry: {entry.path}")
                    if split == "train":
                        require(IMAGENET_TRAIN.fullmatch(entry.name) and entry.name.startswith(wnid + "_"),
                                f"Invalid ImageNet train filename: {entry.path}")
                    else:
                        match = IMAGENET_VAL.fullmatch(entry.name)
                        require(match is not None, f"Invalid ImageNet val filename: {entry.path}")
                        number = int(match.group(1))
                        require(1 <= number <= 50000 and number not in val_ids,
                                f"Duplicate or out-of-range ImageNet val ID: {entry.path}")
                        val_ids.add(number)
                    count += 1
            if split == "val":
                require(count == 50, f"ImageNet val class {wnid} has {count} images, expected 50")
            else:
                require(count > 0, f"Empty ImageNet train class: {wnid}")
            total += count
        totals[split] = total
        require(total == manifest["counts"][split],
                f"ImageNet {split} contains {total} images, expected {manifest['counts'][split]}")
    require(len(val_ids) == 50000, "ImageNet validation IDs are incomplete")
    print("ImageNet-1K train/val verified (structure and counts; source hashes recorded)")
    print(f"  train={totals['train']}, val={totals['val']}, classes={len(classes)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    coco = sub.add_parser("coco", help="check COCO single")
    coco.add_argument("root", type=Path)
    coco.add_argument("--full", action="store_true", help="hash every image and mask")
    waterbirds = sub.add_parser("waterbirds", help="check Waterbirds")
    waterbirds.add_argument("root", type=Path)
    waterbirds.add_argument("--seg-root", type=Path, help="also require CUB segmentation masks")
    imagenet = sub.add_parser("imagenet", help="check ImageNet-1K train and val")
    imagenet.add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "coco":
            check_coco(args.root, args.full)
        elif args.command == "waterbirds":
            check_waterbirds(args.root, args.seg_root)
        elif args.command == "imagenet":
            check_imagenet(args.root)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Asset check failed: {exc}\n")


if __name__ == "__main__":
    main()
