"""Zero-shot classification evaluation for CIFAR test sets."""

from pathlib import Path

import torch
from torchvision.datasets import CIFAR10, CIFAR100

from .common import encode_image_batch, encode_text_batch
from .reporting import plot_confusion_matrix, write_json

PROMPT_TEMPLATE = "a photo of the {}."


def _image_batches(dataset, batch_size):
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        images, labels = zip(*batch)
        yield list(images), torch.tensor(labels, dtype=torch.long)


def _accumulate_statistics(confusion_matrix, scores, labels, include_top5):
    labels = labels.to(device=scores.device, dtype=torch.long)
    predictions = scores.argmax(dim=1)
    class_count = confusion_matrix.shape[0]
    confusion_matrix += torch.bincount(
        labels * class_count + predictions, minlength=class_count * class_count
    ).reshape(class_count, class_count)
    if not include_top5:
        return 0
    top5 = scores.topk(k=min(5, class_count), dim=1).indices
    return (top5 == labels[:, None]).any(dim=1).sum()


def _per_class_accuracy(class_names, class_counts, class_correct):
    return {
        class_name: {
            "num_images": int(class_counts[index].item()),
            "num_correct": int(class_correct[index].item()),
            "accuracy": (
                float(class_correct[index].item() * 100 / class_counts[index].item())
                if class_counts[index].item()
                else None
            ),
        }
        for index, class_name in enumerate(class_names)
    }


@torch.inference_mode()
def run_zero_shot(
    model,
    processor,
    metadata,
    dataset_name,
    data_root,
    output_dir,
    batch_size,
    device,
    download=False,
):
    """Evaluate a checkpoint's image and text encoders on a CIFAR test set."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    if dataset_name == "cifar10":
        dataset = CIFAR10(root=data_root / dataset_name, train=False, download=download)
    elif dataset_name == "cifar100":
        dataset = CIFAR100(root=data_root / dataset_name, train=False, download=download)
    else:
        raise ValueError("dataset_name must be 'cifar10' or 'cifar100'")

    class_names = list(dataset.classes)
    prompts = [PROMPT_TEMPLATE.format(class_name.replace("_", " ")) for class_name in class_names]
    text_features = torch.cat(
        [
            encode_text_batch(model, prompts[start : start + batch_size], device)
            for start in range(0, len(prompts), batch_size)
        ]
    )

    class_count = len(class_names)
    confusion_matrix = torch.zeros((class_count, class_count), dtype=torch.long, device=device)
    top5_correct = 0
    include_top5 = dataset_name == "cifar100"

    for images, labels in _image_batches(dataset, batch_size):
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        for image in images:
            image.close()
        image_features = encode_image_batch(model, pixel_values)
        scores = image_features @ text_features.T
        top5_correct += _accumulate_statistics(confusion_matrix, scores, labels, include_top5)

    confusion_matrix = confusion_matrix.cpu()
    class_counts = confusion_matrix.sum(dim=1)
    class_correct = confusion_matrix.diagonal()
    num_images = int(class_counts.sum().item())
    metrics = {"Top1": float(class_correct.sum().item() * 100 / num_images)}
    if include_top5:
        metrics["Top5"] = int(top5_correct) * 100 / num_images

    report = {
        "metadata": dict(metadata),
        "dataset": dataset_name,
        "split": "test",
        "num_images": num_images,
        "num_classes": class_count,
        "class_names": class_names,
        "prompts": prompts,
        "metric_unit": "percent",
        "metrics": metrics,
        "per_class_accuracy": _per_class_accuracy(class_names, class_counts, class_correct),
        "confusion_matrix": confusion_matrix.tolist(),
    }

    result_dir = output_dir / "zero_shot"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = Path(metadata["checkpoint"]).stem
    write_json(result_dir / f"{dataset_name}_{checkpoint_name}.json", report)
    plot_confusion_matrix(
        confusion_matrix,
        class_names,
        result_dir / f"{dataset_name}_{checkpoint_name}_confusion.png",
    )
    return report
