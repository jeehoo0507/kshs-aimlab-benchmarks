"""The published COCO-single and Waterbirds splits, with no Waterbirds masks."""
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def safe_path(root, relative):
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe dataset path: {relative}")
    return Path(root) / path


class BenchmarkDataset(Dataset):
    def __init__(self, name, root, split, training=False):
        self.name, self.root, self.split, self.training = name, Path(root), split, training
        if name == "coco":
            manifest = json.loads((self.root / "manifest.json").read_text())
            if manifest.get("selection") != "one_annotated_instance_per_image":
                raise ValueError("COCO requires the strict one-object manifest")
            self.classes = manifest["classes"]
            self.records = [r for r in manifest["images"] if (r.get("probe") if split == "probe" else r["split"] == split)]
        elif name == "waterbirds":
            self.classes = ["landbird", "waterbird"]
            with (self.root / "metadata.csv").open(newline="", encoding="utf-8") as stream:
                all_rows = list(csv.DictReader(stream))
            split_index = {"train": 0, "val": 1, "probe": 1, "test": 2}[split]
            self.records = [r for r in all_rows if int(r["split"]) == split_index]
            if split == "probe":
                # 20 fixed validation images per group, independent of training seeds.
                import numpy as np
                rng = np.random.default_rng(20260927)
                selected = []
                for group in range(4):
                    rows = [r for r in self.records if 2 * int(r["y"]) + int(r["place"]) == group]
                    if len(rows) < 20:
                        raise ValueError(f"Waterbirds validation group {group} has fewer than 20 examples")
                    chosen = rng.choice(len(rows), 20, replace=False)
                    selected.extend(rows[int(i)] for i in sorted(chosen))
                self.records = selected
        else:
            raise ValueError(name)
        if not self.records:
            raise ValueError(f"Empty {name}/{split} split")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        relative = row["image"] if self.name == "coco" else row["img_filename"]
        with Image.open(safe_path(self.root, relative)) as source:
            image = source.convert("RGB")
        if self.name == "coco":
            image = TF.resize(image, [224, 224], InterpolationMode.BICUBIC, antialias=True)
            if self.training and torch.rand(()).item() < 0.5:
                image = TF.hflip(image)
            label, group = int(row["label"]), -1
        else:
            if self.training:
                box = RandomResizedCrop.get_params(image, scale=(0.7, 1.0), ratio=(0.75, 4 / 3))
                image = TF.resized_crop(image, *box, [224, 224], InterpolationMode.BICUBIC, antialias=True)
                if torch.rand(()).item() < 0.5:
                    image = TF.hflip(image)
            else:
                image = TF.center_crop(TF.resize(image, 256, InterpolationMode.BICUBIC, antialias=True), [224, 224])
            label, group = int(row["y"]), 2 * int(row["y"]) + int(row["place"])
        item = {"image": TF.normalize(TF.to_tensor(image), MEAN, STD),
                "label": label, "group": group}
        if self.name == "coco" and self.split == "probe":
            with Image.open(safe_path(self.root, row["mask"])) as source:
                mask = source.convert("L")
            mask = TF.resize(mask, [224, 224], InterpolationMode.NEAREST)
            item["foreground"] = torch.nn.functional.avg_pool2d(TF.to_tensor(mask), 16, 16).flatten() >= 0.5
        return item


def loader(dataset, batch_size, training=False):
    # Workers=0 avoids multiplied CPU/RAM usage when seeds run in parallel.
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=0,
                      pin_memory=torch.cuda.is_available(), drop_last=False)
