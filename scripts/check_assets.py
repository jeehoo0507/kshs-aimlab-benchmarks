#!/usr/bin/env python3
"""Validate COCO single and Waterbirds datasets without GPU dependencies."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


CLASSES = ("giraffe", "airplane", "clock", "zebra", "train", "bird",
           "elephant", "toilet", "cow", "bear")
SPLITS = {"train", "val", "test"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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
    require(tuple(manifest.get("classes", [])) == CLASSES, "COCO class order differs from documented definition")
    require(manifest.get("split_seed") == 20260922, "COCO split seed differs from documented definition")
    rows = manifest.get("images")
    require(isinstance(rows, list) and rows, "COCO manifest has no images")
    counts, probes, ids, decoded = Counter(), Counter(), set(), set()
    for row in rows:
        sample_id = row["id"]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    coco = sub.add_parser("coco", help="check COCO single")
    coco.add_argument("root", type=Path)
    coco.add_argument("--full", action="store_true", help="hash every image and mask")
    waterbirds = sub.add_parser("waterbirds", help="check Waterbirds")
    waterbirds.add_argument("root", type=Path)
    waterbirds.add_argument("--seg-root", type=Path, help="also require CUB segmentation masks")
    args = parser.parse_args()
    try:
        if args.command == "coco":
            check_coco(args.root, args.full)
        elif args.command == "waterbirds":
            check_waterbirds(args.root, args.seg_root)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Asset check failed: {exc}\n")


if __name__ == "__main__":
    main()
