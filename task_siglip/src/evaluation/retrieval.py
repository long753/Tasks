"""Complete-pool image-text retrieval evaluation."""

import random
from pathlib import Path

import torch
from tqdm import tqdm

from src.data import ImageCaptionDataset

from .common import encode_image_batch, encode_texts
from .reporting import write_json, write_visualizations


@torch.inference_mode()
def encode_images(model, dataset, processor, batch_size, device):
    """Encode every manifest image in batches while retaining CPU features."""
    features = []
    for start in tqdm(range(0, len(dataset), batch_size), desc="Images", unit="batch"):
        images = [dataset[index]["image"] for index in range(start, min(start + batch_size, len(dataset)))]
        try:
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        finally:
            for image in images:
                image.close()
        features.append(encode_image_batch(model, pixels).cpu())
    return torch.cat(features)


@torch.inference_mode()
def compute_recall(scores: torch.Tensor, text_to_image, ks=(1, 5, 10)):
    """Return hit-rate percentages; scores has image rows and caption columns."""
    owners = torch.as_tensor(text_to_image, dtype=torch.long, device=scores.device)
    max_k = max(ks)
    image_top = scores.topk(min(max_k, scores.shape[1]), dim=1).indices
    text_top = scores.T.topk(min(max_k, scores.shape[0]), dim=1).indices
    image_hits = owners[image_top] == torch.arange(scores.shape[0], device=scores.device)[:, None]
    text_hits = text_top == owners[:, None]
    return {
        "image_to_text": {f"R@{k}": image_hits[:, :k].any(dim=1).float().mean().item() * 100 for k in ks},
        "text_to_image": {f"R@{k}": text_hits[:, :k].any(dim=1).float().mean().item() * 100 for k in ks},
    }


def collect_captions(dataset):
    """Keep every caption's position so equal text from different images stays distinct."""
    texts, text_to_image, caption_records, captions_by_image = [], [], [], []
    for image_index, record in enumerate(dataset.items):
        image_captions = []
        for caption_index, caption in enumerate(record["captions"]):
            text_index = len(texts)
            texts.append(caption)
            text_to_image.append(image_index)
            caption_records.append(
                {
                    "text_index": text_index,
                    "image_index": image_index,
                    "caption_index": caption_index,
                    "text": caption,
                }
            )
            image_captions.append(text_index)
        captions_by_image.append(image_captions)
    return texts, text_to_image, caption_records, captions_by_image


def select_examples(captions_by_image, num_examples, seed):
    """Select distinct images and one of each image's real captions reproducibly."""
    rng = random.Random(seed)
    image_indices = rng.sample(range(len(captions_by_image)), k=min(num_examples, len(captions_by_image)))
    return [(image_index, rng.choice(captions_by_image[image_index])) for image_index in image_indices]


def image_reference(dataset, image_index):
    record = dataset.items[image_index]
    return {
        "image_index": image_index,
        "image_id": record.get("image_id"),
        "image_path": record["image_path"],
    }


def caption_reference(caption_records, text_index):
    return dict(caption_records[text_index])


def build_visualization_samples(scores, dataset, caption_records, text_to_image, selected_examples, top_k=5):
    """Build retrieval results from the full score matrix for the selected queries."""
    owners = [int(owner) for owner in text_to_image]
    image_count, text_count = scores.shape
    image_limit, text_limit = min(top_k, image_count), min(top_k, text_count)
    samples = []
    for image_index, text_index in selected_examples:
        image_top = scores[image_index].topk(text_limit).indices.tolist()
        text_ranking = torch.argsort(scores[:, text_index], descending=True).tolist()
        text_top = text_ranking[:image_limit]
        ground_truth_rank = text_ranking.index(image_index) + 1

        samples.append(
            {
                "image": image_reference(dataset, image_index),
                "caption": caption_reference(caption_records, text_index),
                "image_to_text": {
                    "top5": [
                        {
                            "rank": rank,
                            "cosine": float(scores[image_index, candidate].item()),
                            "is_ground_truth": owners[candidate] == image_index,
                            "caption": caption_reference(caption_records, candidate),
                        }
                        for rank, candidate in enumerate(image_top, 1)
                    ],
                    "ground_truth_captions": [
                        {
                            **caption_reference(caption_records, candidate),
                            "cosine": float(scores[image_index, candidate].item()),
                        }
                        for candidate, owner in enumerate(owners)
                        if owner == image_index
                    ],
                },
                "text_to_image": {
                    "top5": [
                        {
                            "rank": rank,
                            "cosine": float(scores[candidate, text_index].item()),
                            "is_ground_truth": candidate == image_index,
                            "image": image_reference(dataset, candidate),
                        }
                        for rank, candidate in enumerate(text_top, 1)
                    ],
                    "ground_truth": {
                        "image": image_reference(dataset, image_index),
                        "rank": ground_truth_rank,
                        "cosine": float(scores[image_index, text_index].item()),
                    },
                },
            }
        )
    return samples


def run_retrieval(
    model,
    processor,
    metadata,
    manifest_path: Path,
    project_root: Path,
    output_dir: Path,
    batch_size,
    device,
    num_examples=20,
    seed=42,
):
    """Evaluate a checkpoint against all images and captions in a manifest."""
    manifest_path = Path(manifest_path)
    project_root = Path(project_root)
    output_dir = Path(output_dir)

    dataset = ImageCaptionDataset(manifest_path, root=project_root, random_caption=False)
    texts, text_to_image, caption_records, captions_by_image = collect_captions(dataset)
    image_features = encode_images(model, dataset, processor, batch_size, device)
    text_features = encode_texts(model, texts, batch_size, device)

    scores = image_features @ text_features.T
    metrics = compute_recall(scores, text_to_image)
    checkpoint_stem = Path(metadata["checkpoint"]).stem
    report = {
        "checkpoint": metadata["checkpoint"],
        "model": metadata["model"],
        "epoch": metadata["epoch"],
        "manifest": str(manifest_path),
        "num_images": len(dataset),
        "num_texts": len(texts),
        "metric_unit": "percent",
        "metrics": metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{manifest_path.stem}_{checkpoint_stem}.json", report)
    if num_examples:
        selected_examples = select_examples(captions_by_image, num_examples, seed)
        samples = build_visualization_samples(scores, dataset, caption_records, text_to_image, selected_examples)
        write_visualizations(
            dataset,
            samples,
            output_dir / "visualizations" / f"{manifest_path.stem}_{checkpoint_stem}",
            seed,
            len(texts),
        )
    return report
