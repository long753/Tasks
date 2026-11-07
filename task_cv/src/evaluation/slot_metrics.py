"""Permutation-invariant instance-segmentation metrics for Slot Attention."""

from __future__ import annotations

from collections import Counter

import torch
from torch import Tensor


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def segmentation_metrics(predicted: Tensor, target: Tensor) -> dict[str, float]:
    """Measure foreground clustering and matched object masks per image.

    ``foreground_ari`` ignores target-background pixels, but does *not* reserve
    any predicted assignment as background. ``matched_iou`` compares each target
    object with full-image predicted masks, so predicted pixels on background are
    included in the union and count as false positives. Scores are averaged over
    images, not over all pixels or all objects in the batch.

    An image with no target foreground has both scores defined as ``1.0``: there
    is no foreground partition or object match to disagree with. ARI also follows
    the standard convention of returning ``1.0`` when both partitions are
    trivial.
    """
    _validate_inputs(predicted, target)
    predicted_cpu = predicted.detach().to(device="cpu")
    target_cpu = target.detach().to(device="cpu")
    if torch.any(target_cpu < 0):
        raise ValueError("target labels must be nonnegative, with zero as background")

    foreground_ari = 0.0
    matched_iou = 0.0
    for predicted_image, target_image in zip(predicted_cpu, target_cpu):
        foreground_ari += _foreground_ari(predicted_image, target_image)
        matched_iou += _matched_foreground_iou(predicted_image, target_image)

    image_count = predicted_cpu.shape[0]
    return {
        "foreground_ari": foreground_ari / image_count,
        "matched_iou": matched_iou / image_count,
    }


def _validate_inputs(predicted: Tensor, target: Tensor) -> None:
    if not isinstance(predicted, Tensor) or not isinstance(target, Tensor):
        raise TypeError("predicted and target must be torch tensors")
    if predicted.ndim != 3 or target.ndim != 3:
        raise ValueError("predicted and target must have shape [batch, height, width]")
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted and target must have identical shapes, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if predicted.shape[0] == 0 or predicted.shape[1] == 0 or predicted.shape[2] == 0:
        raise ValueError("predicted and target must contain at least one image pixel")
    if predicted.dtype not in _INTEGER_DTYPES or target.dtype not in _INTEGER_DTYPES:
        raise ValueError("predicted and target must use integer label dtypes")


def _foreground_ari(predicted: Tensor, target: Tensor) -> float:
    foreground = target != 0
    if not torch.any(foreground):
        return 1.0

    predicted_labels = [int(value) for value in predicted[foreground].tolist()]
    target_labels = [int(value) for value in target[foreground].tolist()]
    item_count = len(target_labels)
    if item_count < 2:
        return 1.0

    target_counts = Counter(target_labels)
    predicted_counts = Counter(predicted_labels)
    contingency = Counter(zip(target_labels, predicted_labels))
    agreeing_pairs = sum(_choose_two(count) for count in contingency.values())
    target_pairs = sum(_choose_two(count) for count in target_counts.values())
    predicted_pairs = sum(_choose_two(count) for count in predicted_counts.values())
    total_pairs = _choose_two(item_count)

    # This is the standard ARI formula after multiplying numerator and
    # denominator by 2 * total_pairs, preserving exact integer degeneracy checks.
    numerator = 2 * (agreeing_pairs * total_pairs - target_pairs * predicted_pairs)
    denominator = (target_pairs + predicted_pairs) * total_pairs - 2 * target_pairs * predicted_pairs
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _matched_foreground_iou(predicted: Tensor, target: Tensor) -> float:
    target_ids = [int(value) for value in torch.unique(target[target != 0]).tolist()]
    if not target_ids:
        return 1.0

    slot_ids_tensor, slot_counts_tensor = torch.unique(predicted, return_counts=True)
    slot_ids = [int(value) for value in slot_ids_tensor.tolist()]
    slot_counts = {
        slot_id: int(count)
        for slot_id, count in zip(slot_ids, slot_counts_tensor.tolist())
    }
    slot_masks = {slot_id: predicted == slot_id for slot_id in slot_ids}

    ious: list[list[float]] = []
    for target_id in target_ids:
        target_mask = target == target_id
        target_count = int(target_mask.sum().item())
        object_ious: list[float] = []
        for slot_id in slot_ids:
            intersection = int((target_mask & slot_masks[slot_id]).sum().item())
            union = target_count + slot_counts[slot_id] - intersection
            object_ious.append(intersection / union)
        ious.append(object_ious)

    return _best_one_to_one_iou(ious) / len(target_ids)


def _best_one_to_one_iou(ious: list[list[float]]) -> float:
    """Return the exact assignment score, leaving unmatched objects at zero.

    An object may remain unmatched, which is equivalent to assigning one of the
    zero-IoU dummy slots needed when fewer predicted IDs than target objects are
    present. Every real predicted ID may be assigned only once.
    """
    object_count = len(ious)
    best_by_assignment = {0: 0.0}
    for slot_index in range(len(ious[0])):
        next_best = dict(best_by_assignment)
        for assigned, score in best_by_assignment.items():
            for object_index in range(object_count):
                object_bit = 1 << object_index
                if assigned & object_bit:
                    continue
                updated = assigned | object_bit
                candidate = score + ious[object_index][slot_index]
                if candidate > next_best.get(updated, float("-inf")):
                    next_best[updated] = candidate
        best_by_assignment = next_best
    return max(best_by_assignment.values())


def _choose_two(count: int) -> int:
    return count * (count - 1) // 2
