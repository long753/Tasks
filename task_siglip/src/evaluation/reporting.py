"""Output helpers for retrieval and classification evaluation."""

import html
import json
from pathlib import Path

import matplotlib
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_json(path: Path, report) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_thumbnail(dataset, image_index, images_dir):
    thumbnail_path = images_dir / f"{image_index}.jpg"
    if not thumbnail_path.exists():
        record = dataset.items[image_index]
        with Image.open(dataset.root / record["image_path"]) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((480, 480))
            thumbnail.save(thumbnail_path, "JPEG", quality=90)
    return thumbnail_path.relative_to(images_dir.parent).as_posix()


def add_thumbnails(dataset, samples, images_dir):
    for sample in samples:
        images = [sample["image"], sample["text_to_image"]["ground_truth"]["image"]]
        images.extend(result["image"] for result in sample["text_to_image"]["top5"])
        for image in images:
            image["thumbnail"] = write_thumbnail(dataset, image["image_index"], images_dir)


def describe_image(image):
    identifier = f"image {image['image_index']}"
    if image["image_id"] is not None:
        identifier += f" ({image['image_id']})"
    return f"{identifier}: {image['image_path']}"


def describe_caption(caption):
    return f"caption {caption['caption_index']} of image {caption['image_index']}"


def render_image(image, class_name):
    label = describe_image(image)
    thumbnail = html.escape(image["thumbnail"], quote=True)
    return (
        f'<figure class="{class_name}"><img src="{thumbnail}" alt="{html.escape(label, quote=True)}">'
        f"<figcaption>{html.escape(label)}</figcaption></figure>"
    )


def render_caption(caption):
    return (
        f"<p><strong>{html.escape(describe_caption(caption))}</strong></p>"
        f'<p class="caption">{html.escape(caption["text"])}</p>'
    )


def render_visualization(document):
    lines = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>SigLIP retrieval examples</title>",
        """<style>
body { font-family: system-ui, sans-serif; line-height: 1.45; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }
article { border-top: 2px solid #333; margin-top: 2rem; padding-top: 1rem; }
section { margin: 1.5rem 0; }
ol { padding-left: 1.5rem; }
li { margin: .75rem 0; }
figure { margin: .5rem 0; }
.query-image img { display: block; max-height: 480px; max-width: 100%; object-fit: contain; }
.result-image img { display: block; height: 160px; max-width: 100%; object-fit: contain; }
figcaption, .caption { max-width: 75ch; overflow-wrap: anywhere; white-space: pre-wrap; }
.image-results { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }
.image-results li { min-width: 0; }
.match { font-weight: 700; text-decoration: underline; background: #e7f6ec; padding: .1rem .3rem; }
</style></head><body>""",
        "<h1>SigLIP retrieval examples</h1>",
        (
            f"<p>Seed: {document['seed']}. Candidate pool: "
            f"{document['candidate_pool']['num_images']} images and "
            f"{document['candidate_pool']['num_captions']} captions.</p>"
        ),
    ]
    for number, sample in enumerate(document["samples"], 1):
        lines.extend(
            [
                f"<article><h2>Example {number}</h2>",
                "<h3>Query image</h3>",
                render_image(sample["image"], "query-image"),
                "<section><h3>Image → text: top candidates</h3><ol>",
            ]
        )
        for result in sample["image_to_text"]["top5"]:
            marker = '<span class="match">Ground-truth match</span>' if result["is_ground_truth"] else "Not a match"
            lines.append(
                f"<li><strong>Rank {result['rank']}</strong>; cosine {result['cosine']:.4f}; {marker}"
                f"{render_caption(result['caption'])}</li>"
            )
        lines.extend(["</ol><h4>All ground-truth descriptions for this image</h4><ol>"])
        for caption in sample["image_to_text"]["ground_truth_captions"]:
            lines.append(f"<li>cosine {caption['cosine']:.4f}{render_caption(caption)}</li>")
        lines.extend(
            [
                "</ol></section>",
                "<section><h3>Text → image</h3><h4>Query text</h4>",
                render_caption(sample["caption"]),
                '<h4>Top candidates</h4><ol class="image-results">',
            ]
        )
        for result in sample["text_to_image"]["top5"]:
            marker = (
                '<span class="match">Ground-truth image</span>'
                if result["is_ground_truth"]
                else "Not the ground-truth image"
            )
            lines.append(
                f"<li><strong>Rank {result['rank']}</strong>; cosine {result['cosine']:.4f}; {marker}"
                f"{render_image(result['image'], 'result-image')}</li>"
            )
        ground_truth = sample["text_to_image"]["ground_truth"]
        lines.extend(
            [
                "</ol><h4>Ground-truth image in the complete candidate pool</h4>",
                (
                    f"<p><strong>Rank {ground_truth['rank']}</strong>; cosine {ground_truth['cosine']:.4f}; "
                    "Ground-truth image</p>"
                ),
                render_image(ground_truth["image"], "result-image"),
                "</section></article>",
            ]
        )
    lines.append("</body></html>")
    return "\n".join(lines) + "\n"


def write_visualizations(dataset, samples, visualization_dir, seed, num_captions):
    images_dir = visualization_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    add_thumbnails(dataset, samples, images_dir)
    document = {
        "seed": seed,
        "candidate_pool": {"num_images": len(dataset), "num_captions": num_captions},
        "samples": samples,
    }
    write_json(visualization_dir / "samples.json", document)
    html_path = visualization_dir / "index.html"
    html_path.write_text(render_visualization(document), encoding="utf-8")
    return html_path


def plot_confusion_matrix(confusion_matrix, class_names, output_path: Path) -> None:
    """Save a row-normalized confusion-matrix image without changing source counts."""
    counts = torch.as_tensor(confusion_matrix).detach().cpu()
    normalized = counts.to(dtype=torch.float64) * 100 / counts.sum(dim=1, keepdim=True).clamp_min(1)

    figure_size = max(8, len(class_names) * 0.2)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(normalized.numpy(), interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized percentage")
    indices = range(len(class_names))
    axis.set_xticks(indices, labels=class_names, rotation=90, fontsize=6)
    axis.set_yticks(indices, labels=class_names, fontsize=6)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Confusion matrix (rows normalized; raw counts are in the JSON report)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
