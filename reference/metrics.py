"""Accuracy and frozen-validation-probe diagnostics (values are fractions)."""
import numpy as np
import torch
from torch.nn import functional as F

from .data import loader


def autocast(device):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")


@torch.inference_mode()
def evaluate(model, data, device, batch_size):
    model.eval()
    counts = [0] * len(data.classes)
    correct = [0] * len(data.classes)
    group_counts, group_correct = [0] * 4, [0] * 4
    for batch in loader(data, batch_size):
        with autocast(device):
            prediction = model(batch["image"].to(device)).argmax(1).cpu()
        for label, group, pred in zip(batch["label"].tolist(), batch["group"].tolist(), prediction.tolist()):
            counts[label] += 1
            correct[label] += int(pred == label)
            if group >= 0:
                group_counts[group] += 1
                group_correct[group] += int(pred == label)
    if any(count == 0 for count in counts):
        raise ValueError("Evaluation split is missing a class")
    per_class = [c / n for c, n in zip(correct, counts)]
    score = {"n": sum(counts), "correct": sum(correct), "overall_accuracy": sum(correct) / sum(counts),
             "macro_accuracy": sum(per_class) / len(per_class),
             "class_counts": counts, "class_correct": correct, "class_accuracy": per_class}
    if data.name == "waterbirds":
        if any(count == 0 for count in group_counts):
            raise ValueError("Waterbirds evaluation split is missing a group")
        group_accuracy = [c / n for c, n in zip(group_correct, group_counts)]
        score.update(group_counts=group_counts, group_correct=group_correct,
                     group_accuracy=group_accuracy, worst_group_accuracy=min(group_accuracy))
    return score


@torch.inference_mode()
def mask_probe(student, teacher, data, device, batch_size, keep=98):
    student.eval()
    teacher.eval()
    n = full_correct = masked_correct = full_to_wrong = 0
    kl_sum = fg_recall_sum = fg_precision_sum = 0.0
    fg_images = 0
    selection_hex = []
    for batch in loader(data, batch_size):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with autocast(device):
            _, attention = student(images, return_attention=True)
            indices = attention.topk(keep, dim=1).indices
            full_logits = teacher(images).float()
            masked_logits = teacher(images, indices).float()
        full_prediction, masked_prediction = full_logits.argmax(1), masked_logits.argmax(1)
        is_full_correct = full_prediction.eq(labels)
        full_correct += is_full_correct.sum().item()
        masked_correct += masked_prediction.eq(labels).sum().item()
        full_to_wrong += (is_full_correct & masked_prediction.ne(labels)).sum().item()
        kl = F.kl_div(F.log_softmax(masked_logits, dim=1), F.softmax(full_logits, dim=1),
                      reduction="none").sum(dim=1)
        kl_sum += kl.sum().item()
        selected = torch.zeros(len(images), 196, dtype=torch.bool, device=device)
        selected.scatter_(1, indices, True)
        packed = np.packbits(selected.cpu().numpy(), axis=1)
        selection_hex.extend(row.tobytes().hex() for row in packed)
        if "foreground" in batch:
            fg = batch["foreground"].to(device)
            selected_fg = fg.gather(1, indices).sum(dim=1)
            fg_total = fg.sum(dim=1)
            valid = fg_total > 0
            fg_recall_sum += (selected_fg[valid] / fg_total[valid]).sum().item()
            fg_precision_sum += (selected_fg[valid] / keep).sum().item()
            fg_images += valid.sum().item()
        n += len(images)
    return {"n": n, "teacher_full_accuracy": full_correct / n,
            "teacher_masked_accuracy": masked_correct / n,
            "full_correct_masked_wrong": full_to_wrong / n,
            "kl_full_to_masked": kl_sum / n,
            "foreground_recall": fg_recall_sum / fg_images if fg_images else None,
            "foreground_precision": fg_precision_sum / fg_images if fg_images else None,
            "foreground_images": fg_images, "selection_hex": selection_hex}


def selection_overlap(current, previous, keep=98):
    if len(current) != len(previous):
        raise ValueError("Probe membership changed")
    shared = 0
    for one, other in zip(current, previous):
        shared += (int.from_bytes(bytes.fromhex(one), "big") &
                   int.from_bytes(bytes.fromhex(other), "big")).bit_count()
    return shared / (len(current) * keep)
