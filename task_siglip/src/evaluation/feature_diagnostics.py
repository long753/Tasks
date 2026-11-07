"""Feature diversity, dimensionality and image-text alignment diagnostics."""

import random
from pathlib import Path

import matplotlib
import torch
from tqdm import tqdm

from src.data import ImageCaptionDataset

from .common import encode_image_batch, encode_text_batch
from .reporting import write_json
from .retrieval import compute_recall

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def select_samples(dataset, num_samples, seed):
    """Select the same image-caption pairs independently of model initialization."""
    if min(num_samples, len(dataset)) < 2:
        raise ValueError("Feature diagnostics require at least two images")
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(num_samples, len(dataset)))
    samples = []
    for index in indices:
        record = dataset.items[index]
        caption_index = rng.randrange(len(record["captions"]))
        samples.append(
            {
                "dataset_index": index,
                "image_id": record["image_id"],
                "image_path": record["image_path"],
                "caption_index": caption_index,
                "text": record["captions"][caption_index],
            }
        )
    if any(len({sample[key] for sample in samples}) != len(samples) for key in ("image_id", "image_path")):
        raise ValueError("Selected records contain duplicate images; use a manifest with one record per image")
    return samples


@torch.inference_mode()
def extract_features(model, dataset, samples, processor, batch_size, device):
    model.eval()
    image_features, text_features = [], []
    for start in tqdm(range(0, len(samples), batch_size), desc="Diagnostic features", unit="batch"):
        batch = samples[start : start + batch_size]
        images = []
        try:
            for sample in batch:
                images.append(dataset[sample["dataset_index"]]["image"])
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        finally:
            for image in images:
                image.close()
        image_features.append(encode_image_batch(model, pixels, normalize=False).float().cpu())
        text_features.append(
            encode_text_batch(model, [sample["text"] for sample in batch], device, normalize=False).float().cpu()
        )
    return torch.cat(image_features), torch.cat(text_features)


def _summary(values):
    if values.numel() == 0:
        return None
    quantiles = torch.quantile(values, values.new_tensor([0.05, 0.5, 0.95]))
    return {
        "count": values.numel(),
        "mean": values.mean().item(),
        "std": values.std(correction=0).item(),
        "min": values.min().item(),
        "p05": quantiles[0].item(),
        "median": quantiles[1].item(),
        "p95": quantiles[2].item(),
        "max": values.max().item(),
    }


def feature_statistics(raw_features):
    """Return metrics, normalized features and off-diagonal cosines on CPU.

    Covariance uses 1/N and unit-length features. Entropy effective rank uses
    covariance eigenvalues (not singular values). Invalid directions are not
    silently dropped or replaced by epsilon-normalized zero vectors.
    """
    features = raw_features.detach().to(device="cpu", dtype=torch.float64)
    count, dimension = features.shape
    if count < 2 or dimension < 1:
        raise ValueError("Expected at least two feature vectors with a nonempty feature dimension")
    finite_rows = torch.isfinite(features).all(dim=1)
    zero_rows = finite_rows & (features == 0).all(dim=1)
    norms = torch.linalg.vector_norm(features[finite_rows], dim=1)
    metrics = {
        "num_samples": count,
        "dimension": dimension,
        "raw_norm": _summary(norms),
        "nonfinite_elements": (~torch.isfinite(features)).sum().item(),
        "nonfinite_vectors": (~finite_rows).sum().item(),
        "zero_vectors": zero_rows.sum().item(),
        "mean_vector_norm": None,
        "total_variance": None,
        "effective_rank": None,
        "effective_rank_ratio": None,
        "max_rank": min(count - 1, dimension),
        "largest_eigenvalue_fraction": None,
        "covariance_eigenvalues": None,
        "off_diagonal_cosine": None,
    }
    if metrics["nonfinite_vectors"] or metrics["zero_vectors"]:
        metrics["status"] = "invalid_directions"
        return metrics, None, None

    # Explicit division also preserves the direction of nonzero vectors smaller
    # than F.normalize's default epsilon. Zero vectors were checked above.
    normalized = features / norms[:, None]
    # Subtracting a reference first makes identical vectors center to exactly zero.
    centered = normalized - normalized[0]
    centered.sub_(centered.mean(dim=0))
    covariance = centered.T @ centered / count
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0).clamp_min_(0)
    eigenvalues[metrics["max_rank"] :] = 0  # The remaining directions have exactly zero theoretical rank.
    total_variance = centered.square().sum().item() / count
    if total_variance == 0:
        effective_rank, largest_fraction = 0.0, None
    else:
        probabilities = eigenvalues / eigenvalues.sum()
        positive = probabilities[probabilities > 0]
        effective_rank = min(torch.exp(-(positive * positive.log()).sum()).item(), metrics["max_rank"])
        largest_fraction = probabilities[0].item()
    cosine = (normalized @ normalized.T).clamp_(-1, 1)
    off_diagonal = cosine[~torch.eye(count, dtype=torch.bool)]
    metrics.update(
        status="constant_directions" if total_variance == 0 else "ok",
        mean_vector_norm=normalized.mean(dim=0).norm().item(),
        total_variance=total_variance,
        effective_rank=effective_rank,
        effective_rank_ratio=effective_rank / metrics["max_rank"],
        largest_eigenvalue_fraction=largest_fraction,
        covariance_eigenvalues=eigenvalues.tolist(),
        off_diagonal_cosine=_summary(off_diagonal),
    )
    return metrics, normalized, off_diagonal


def alignment_statistics(image_features, text_features, seed):
    """Real ownership is diagonal; a separate wrong-label control is a derangement."""
    if image_features is None or text_features is None:
        return {"status": "unavailable", "reason": "Zero or nonfinite feature vectors have undefined directions"}, None
    scores = (image_features @ text_features.T).clamp_(-1, 1)
    count = len(scores)
    indices = torch.arange(count)
    paired = scores.diagonal()
    negatives = scores[~torch.eye(count, dtype=torch.bool)]
    shift = random.Random(seed).randrange(1, count)
    wrong_owners = (indices + shift) % count
    wrong_pairs = scores[wrong_owners, indices]
    metrics = {
        "status": "ok",
        "paired_cosine": _summary(paired),
        "negative_cosine": _summary(negatives),
        "paired_minus_negative_mean": (paired.mean() - negatives.mean()).item(),
        "retrieval_percent": compute_recall(scores, indices),
        "shuffled_control": {
            "method": "wrong image-owner labels via nonzero cyclic shift; not real ground truth",
            "shift": shift,
            "cosine": _summary(wrong_pairs),
            "retrieval_percent": compute_recall(scores, wrong_owners),
        },
    }
    return metrics, (paired, negatives, wrong_pairs)


def _plot_diagnostics(image_metrics, text_metrics, image_cosines, text_cosines, alignment_values, result_dir):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for label, values in (("Image", image_cosines), ("Text", text_cosines)):
        if values is not None:
            axes[0].hist(values.numpy(), bins=60, range=(-1, 1), density=True, histtype="step", label=label)
    axes[0].set(title="Within-modality off-diagonal cosine", xlabel="Cosine similarity", ylabel="Density")
    if image_cosines is not None or text_cosines is not None:
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "Undefined: invalid feature directions", ha="center", transform=axes[0].transAxes)
    if alignment_values is not None:
        for label, values in zip(("Matched", "Other images", "Shuffled-label control"), alignment_values, strict=True):
            axes[1].hist(values.numpy(), bins=60, range=(-1, 1), density=True, histtype="step", label=label)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "Undefined: invalid feature directions", ha="center", transform=axes[1].transAxes)
    axes[1].set(title="Cross-modal cosine", xlabel="Cosine similarity", ylabel="Density")
    figure.savefig(result_dir / "cosine_distributions.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for label, metrics in (("Image", image_metrics), ("Text", text_metrics)):
        eigenvalues = metrics["covariance_eigenvalues"]
        if eigenvalues is not None:
            axis.plot(range(1, len(eigenvalues) + 1), eigenvalues, label=label)
    axis.set(
        title="Centered covariance spectrum (unit-length features)",
        xlabel="Eigenvalue index (descending)",
        ylabel="Eigenvalue",
    )
    axis.set_yscale("symlog", linthresh=1e-10)
    if image_metrics["covariance_eigenvalues"] is not None or text_metrics["covariance_eigenvalues"] is not None:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "Undefined: invalid feature directions", ha="center", transform=axis.transAxes)
    figure.savefig(result_dir / "covariance_spectrum.png", dpi=150)
    plt.close(figure)


@torch.inference_mode()
def run_feature_diagnostics(
    model, processor, metadata, manifest_path, project_root, output_dir, batch_size, device, num_samples=1024, seed=42
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    manifest_path = Path(manifest_path)
    dataset = ImageCaptionDataset(manifest_path, root=project_root, random_caption=False)
    samples = select_samples(dataset, num_samples, seed)
    image_raw, text_raw = extract_features(model, dataset, samples, processor, batch_size, device)
    image_metrics, image_features, image_cosines = feature_statistics(image_raw)
    text_metrics, text_features, text_cosines = feature_statistics(text_raw)
    alignment, alignment_values = alignment_statistics(image_features, text_features, seed)
    if metadata.get("initialization") == "random":
        label = f"random_init_seed{metadata['seed']}"
    else:
        label = Path(metadata["checkpoint"]).stem
    result_dir = Path(output_dir) / "feature_diagnostics" / f"{manifest_path.stem}_{label}"
    result_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": metadata,
        "manifest": str(manifest_path),
        "sampling": {
            "seed": seed,
            "requested_num_samples": num_samples,
            "num_samples": len(samples),
            "samples": samples,
        },
        "definitions": {
            "features": "Encoder outputs before logit_scale/logit_bias; one fixed caption per distinct image",
            "covariance": "Centered unit-length features, covariance divided by N, calculated in float64 on CPU",
            "effective_rank": "exp(entropy of normalized covariance eigenvalues); 0 for zero variance",
            "effective_rank_ratio": "effective_rank / min(N - 1, feature_dimension)",
            "off_diagonal_cosine": "All ordered pairs of distinct samples; excludes self-similarity",
            "shuffled_control": "Wrong ownership labels only; true ownership remains unchanged for real Recall",
            "interpretation": "No universal collapse threshold; inspect variance, spectrum and alignment together",
        },
        "metrics": {"image": image_metrics, "text": text_metrics, "alignment": alignment},
        "artifacts": {
            "report": str(result_dir / "report.json"),
            "cosine_distributions": str(result_dir / "cosine_distributions.png"),
            "covariance_spectrum": str(result_dir / "covariance_spectrum.png"),
        },
    }
    _plot_diagnostics(image_metrics, text_metrics, image_cosines, text_cosines, alignment_values, result_dir)
    write_json(result_dir / "report.json", report)
    return report
