from __future__ import annotations
from torch.nn.parameter import Parameter

import argparse
import json
import sys
from colorsys import hsv_to_rgb
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use(backend="Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(object=PROJECT_ROOT))

from src.data.slot_scenes import RandomShapeScenes
from src.evaluation.slot_metrics import segmentation_metrics
from src.model.slot_attention import SlotAttentionAutoEncoder


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Slot Attention on random scenes or run it on one RGB image."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Input image. Omitting this trains on random synthetic scenes.",
    )
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=10000,
        help="Number of optimizer updates when training (default: 10000).",
    )
    parser.add_argument("--lr", type=positive_float, default=4e-4)
    parser.add_argument("--slot-count", type=positive_int, default=4)
    parser.add_argument(
        "--image-size",
        type=positive_int,
        default=None,
        help="Model image size (default: 64 for scenes, 128 for --image).",
    )
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--val-size", type=positive_int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, or 'auto' to select CUDA when available.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to load for inference-only visualization.",
    )
    return parser.parse_args(args=argv)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def select_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def create_output_dir() -> Path:
    output_root: Path = PROJECT_ROOT / "outputs" / "test"
    while True:
        timestamp: str = datetime.now().strftime(format="%Y%m%d-%H%M%S-%f")
        output_dir: Path = output_root / timestamp
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            return output_dir
        except FileExistsError:
            continue


def load_image(
    image_path: Path,
    image_size: int,
    model: SlotAttentionAutoEncoder,
) -> torch.Tensor:
    preprocess = transforms.Compose(
        transforms=[
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    with Image.open(fp=image_path) as source:
        image = preprocess(img=source.convert(mode="RGB")).unsqueeze(0)

    parameter: Parameter = next(model.parameters())
    return image.to(device=parameter.device, dtype=parameter.dtype)


def train_image(
    model: SlotAttentionAutoEncoder,
    image: torch.Tensor,
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for step in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image)
        loss = F.mse_loss(outputs["reconstruction"], image)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite reconstruction loss at step {step}: {loss.item()}"
            )
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == epochs:
            print(
                f"step {step}/{epochs}: reconstruction_mse={loss.item():.8f}",
                flush=True,
            )


def evaluate_image(
    model: SlotAttentionAutoEncoder,
    image: torch.Tensor,
    seed: int,
) -> tuple[dict[str, torch.Tensor], float]:
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        reconstruction_mse = F.mse_loss(outputs["reconstruction"], image).item()
    return outputs, reconstruction_mse


def evaluate_synthetic(
    model: SlotAttentionAutoEncoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    model.eval()
    parameter = next(model.parameters())

    total_mse = 0.0
    total_foreground_ari = 0.0
    total_matched_iou = 0.0
    visual_outputs: dict[str, list[torch.Tensor]] = {}
    visual_count = 0

    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            end = min(start + batch_size, images.shape[0])
            batch_images = images[start:end].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            outputs = model(batch_images)
            batch_count = end - start
            total_mse += (
                F.mse_loss(outputs["reconstruction"], batch_images).item() * batch_count
            )

            assignments = outputs["masks"].argmax(dim=1).squeeze(1)
            batch_metrics = segmentation_metrics(assignments, labels[start:end])
            total_foreground_ari += batch_metrics["foreground_ari"] * batch_count
            total_matched_iou += batch_metrics["matched_iou"] * batch_count

            if visual_count < 4:
                take = min(4 - visual_count, batch_count)
                for name, value in outputs.items():
                    visual_outputs.setdefault(name, []).append(value[:take].detach().cpu())
                visual_count += take

    image_count = images.shape[0]
    return (
        {
            "reconstruction_mse": total_mse / image_count,
            "foreground_ari": total_foreground_ari / image_count,
            "matched_iou": total_matched_iou / image_count,
        },
        {name: torch.cat(parts, dim=0) for name, parts in visual_outputs.items()},
    )


def train_synthetic(
    model: SlotAttentionAutoEncoder,
    scenes: RandomShapeScenes,
    validation_images: torch.Tensor,
    validation_labels: torch.Tensor,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    parameter = next(model.parameters())
    last_validation: tuple[dict[str, float], dict[str, torch.Tensor]] | None = None
    model.train()

    for step in range(1, epochs + 1):
        batch_images, _ = scenes.sample(batch_size)
        batch_images = batch_images.to(device=parameter.device, dtype=parameter.dtype)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_images)
        loss = F.mse_loss(outputs["reconstruction"], batch_images)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite reconstruction loss at step {step}: {loss.item()}"
            )
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == epochs:
            print(
                f"step {step}/{epochs}: reconstruction_mse={loss.item():.8f}",
                flush=True,
            )

        if step == 1 or step % 500 == 0 or step == epochs:
            last_validation = evaluate_synthetic(
                model,
                validation_images,
                validation_labels,
                batch_size,
                seed,
            )
            metrics = last_validation[0]
            print(
                "validation "
                f"step {step}/{epochs}: mse={metrics['reconstruction_mse']:.8f} "
                f"fg_ari={metrics['foreground_ari']:.6f} "
                f"matched_iou={metrics['matched_iou']:.6f}",
                flush=True,
            )
            model.train()

    if last_validation is None:
        raise RuntimeError("Synthetic training completed without validation.")
    return last_validation


def save_feature_maps(
    tokens: torch.Tensor,
    image_size: int,
    output_path: Path,
) -> None:
    feature_size = image_size // 4
    feature_maps = tokens[0].transpose(0, 1).reshape(
        tokens.shape[-1], 1, feature_size, feature_size
    )
    save_image(
        feature_maps,
        output_path,
        nrow=8,
        normalize=True,
        scale_each=True,
    )


def slot_palette(slot_count: int) -> torch.Tensor:
    return torch.tensor(
        [hsv_to_rgb(slot / slot_count, 0.65, 0.95) for slot in range(slot_count)]
    )


def colorize_instance_labels(labels: torch.Tensor) -> torch.Tensor:
    object_count = int(labels.max().item())
    palette = torch.zeros(object_count + 1, 3)
    if object_count:
        palette[1:] = torch.tensor(
            [
                hsv_to_rgb(((object - 1) / object_count + 0.11) % 1.0, 0.65, 0.95)
                for object in range(1, object_count + 1)
            ]
        )
    return palette[labels]


def save_segmentation(
    image: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    reconstruction_mse: float,
    output_path: Path,
    target_labels: torch.Tensor | None = None,
) -> None:
    reconstruction = outputs["reconstruction"][0].detach().cpu()
    masks = outputs["masks"][0, :, 0].detach().cpu()
    slot_reconstructions = outputs["slot_reconstructions"][0].detach().cpu()
    original = image[0].detach().cpu()
    slot_count = masks.shape[0]
    palette = slot_palette(slot_count)

    assignments = masks.argmax(dim=0)
    overlay_colors = palette[assignments]
    original_rgb = original.permute(1, 2, 0).numpy()
    overlay = 0.55 * original_rgb + 0.45 * overlay_colors.numpy()
    columns = max(4 if target_labels is not None else 3, slot_count)
    figure, axes = plt.subplots(3, columns, figsize=(3 * columns, 9), squeeze=False)

    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(reconstruction.permute(1, 2, 0).numpy().clip(0, 1))
    axes[0, 1].set_title("Reconstruction")
    axes[0, 2].imshow(overlay.clip(0, 1))
    axes[0, 2].set_title("Argmax mask overlay")
    if target_labels is not None:
        axes[0, 3].imshow(colorize_instance_labels(target_labels.detach().cpu()).numpy())
        axes[0, 3].set_title("True instance labels")

    area_fractions = masks.mean(dim=(1, 2))
    for slot in range(slot_count):
        color = palette[slot].tolist()
        axes[1, slot].imshow(masks[slot].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, slot].set_title(
            f"Slot {slot + 1} mask\narea {area_fractions[slot].item():.1%}",
            color=color,
        )
        contribution = slot_reconstructions[slot] * masks[slot].unsqueeze(0)
        axes[2, slot].imshow(contribution.permute(1, 2, 0).numpy().clip(0, 1))
        axes[2, slot].set_title(f"Slot {slot + 1} contribution", color=color)

    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(f"Reconstruction MSE: {reconstruction_mse:.8f}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_validation(
    images: torch.Tensor,
    labels: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    validation_metrics: dict[str, float],
    output_path: Path,
) -> None:
    example_count = min(4, images.shape[0])
    figure, axes = plt.subplots(
        example_count,
        4,
        figsize=(12, 3 * example_count),
        squeeze=False,
    )
    palette = slot_palette(outputs["masks"].shape[1])

    for index in range(example_count):
        original_rgb = images[index].detach().cpu().permute(1, 2, 0).numpy()
        target_labels = labels[index].detach().cpu()
        assignments = outputs["masks"][index, :, 0].detach().cpu().argmax(dim=0)
        overlay_colors = palette[assignments].numpy()
        overlay = 0.55 * original_rgb + 0.45 * overlay_colors
        reconstruction = outputs["reconstruction"][index].detach().cpu()
        per_image_metrics = segmentation_metrics(
            assignments.unsqueeze(0),
            target_labels.unsqueeze(0),
        )

        axes[index, 0].imshow(original_rgb)
        axes[index, 0].set_title(f"Held-out {index + 1}: original")
        axes[index, 1].imshow(colorize_instance_labels(target_labels).numpy())
        axes[index, 1].set_title("Ground-truth instances")
        axes[index, 2].imshow(overlay.clip(0, 1))
        axes[index, 2].set_title(
            "Predicted overlay\n"
            f"FG-ARI {per_image_metrics['foreground_ari']:.3f}, "
            f"mIoU {per_image_metrics['matched_iou']:.3f}"
        )
        axes[index, 3].imshow(reconstruction.permute(1, 2, 0).numpy().clip(0, 1))
        axes[index, 3].set_title("Reconstruction")

    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(
        "Fixed held-out scenes: "
        f"MSE {validation_metrics['reconstruction_mse']:.6f}, "
        f"FG-ARI {validation_metrics['foreground_ari']:.4f}, "
        f"matched IoU {validation_metrics['matched_iou']:.4f}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_synthetic_metrics(
    output_path: Path,
    model_config: dict[str, int | bool],
    scene_config: dict[str, int],
    seed: int,
    eval_batch_size: int,
    validation_metrics: dict[str, float],
) -> None:
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(
            {
                "model_config": model_config,
                "scene_config": scene_config,
                "seed": seed,
                "eval_batch_size": eval_batch_size,
                "final_validation": validation_metrics,
            },
            destination,
            indent=2,
            sort_keys=True,
        )
        destination.write("\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    checkpoint_path = (
        resolve_project_path(args.checkpoint) if args.checkpoint is not None else None
    )
    checkpoint: dict[str, object] | None = None
    scene_config: dict[str, int] | None = None
    eval_batch_size = args.batch_size

    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=True,
            map_location=device,
        )
        model_config = checkpoint["model_config"]
        seed = checkpoint["seed"]
        saved_image_path = checkpoint["image_path"]
        if args.image is not None:
            mode = "image"
            image_path = resolve_project_path(args.image)
        elif saved_image_path is None:
            mode = "synthetic"
            scene_config = checkpoint["scene_config"]
            eval_batch_size = checkpoint["eval_batch_size"]
            image_path = None
        else:
            mode = "image"
            image_path = resolve_project_path(saved_image_path)
    elif args.image is not None:
        mode = "image"
        seed = args.seed
        model_config = {
            "image_size": args.image_size if args.image_size is not None else 128,
            "slot_count": args.slot_count,
            "slot_dim": 64,
            "num_iterations": 3,
        }
        image_path = resolve_project_path(args.image)
    else:
        if args.slot_count < 2:
            raise ValueError("Synthetic scene training requires --slot-count of at least 2.")
        mode = "synthetic"
        seed = args.seed
        model_config = {
            "image_size": args.image_size if args.image_size is not None else 64,
            "slot_count": args.slot_count,
            "slot_dim": 64,
            "num_iterations": 3,
            "constant_color": True,
        }
        scene_config = {
            "seed": seed,
            "val_count": args.val_size,
            "max_objects": min(3, args.slot_count - 1),
        }
        image_path = None

    if mode == "synthetic" and model_config["slot_count"] < 2:
        raise ValueError("Synthetic scene training requires at least two slots.")

    torch.manual_seed(seed)
    model = SlotAttentionAutoEncoder(**model_config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    output_dir = create_output_dir()

    if mode == "image":
        if image_path is None:
            raise RuntimeError("Image mode requires an image path.")
        image = load_image(image_path, model_config["image_size"], model)
        if checkpoint is None:
            train_image(model, image, args.epochs, args.lr)

        outputs, reconstruction_mse = evaluate_image(model, image, seed)
        save_feature_maps(
            outputs["tokens"],
            model_config["image_size"],
            output_dir / "feature_maps.png",
        )
        save_segmentation(
            image,
            outputs,
            reconstruction_mse,
            output_dir / "segmentation.png",
        )

        if checkpoint is None:
            torch.save(
                {
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "image_path": str(image_path),
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "eval_batch_size": 1,
                    "final_reconstruction_mse": reconstruction_mse,
                },
                output_dir / "model.pt",
            )
        else:
            print(f"input checkpoint: {checkpoint_path}", flush=True)

        print(f"final reconstruction MSE: {reconstruction_mse:.8f}", flush=True)
    else:
        if scene_config is None:
            raise RuntimeError("Synthetic mode requires a scene configuration.")
        scene_seed = scene_config["seed"]
        max_objects = scene_config["max_objects"]
        validation_count = scene_config["val_count"]
        validation_scenes = RandomShapeScenes(
            model_config["image_size"],
            max_objects=max_objects,
            seed=scene_seed + 10000,
        )
        validation_images, validation_labels = validation_scenes.sample(validation_count)

        if checkpoint is None:
            train_scenes = RandomShapeScenes(
                model_config["image_size"],
                max_objects=max_objects,
                seed=scene_seed,
            )
            validation_metrics, visual_outputs = train_synthetic(
                model,
                train_scenes,
                validation_images,
                validation_labels,
                args.epochs,
                args.lr,
                eval_batch_size,
                seed,
            )
        else:
            validation_metrics, visual_outputs = evaluate_synthetic(
                model,
                validation_images,
                validation_labels,
                eval_batch_size,
                seed,
            )
            print(f"input checkpoint: {checkpoint_path}", flush=True)

        save_feature_maps(
            visual_outputs["tokens"],
            model_config["image_size"],
            output_dir / "feature_maps.png",
        )
        save_segmentation(
            validation_images[:1],
            {name: value[:1] for name, value in visual_outputs.items()},
            validation_metrics["reconstruction_mse"],
            output_dir / "segmentation.png",
            validation_labels[0],
        )
        save_validation(
            validation_images[:4],
            validation_labels[:4],
            visual_outputs,
            validation_metrics,
            output_dir / "validation.png",
        )
        save_synthetic_metrics(
            output_dir / "metrics.json",
            model_config,
            scene_config,
            seed,
            eval_batch_size,
            validation_metrics,
        )

        if checkpoint is None:
            torch.save(
                {
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "image_path": None,
                    "scene_config": scene_config,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "eval_batch_size": eval_batch_size,
                    "final_reconstruction_mse": validation_metrics[
                        "reconstruction_mse"
                    ],
                    "final_validation": validation_metrics,
                },
                output_dir / "model.pt",
            )

        print(
            "final validation: "
            f"mse={validation_metrics['reconstruction_mse']:.8f} "
            f"fg_ari={validation_metrics['foreground_ari']:.6f} "
            f"matched_iou={validation_metrics['matched_iou']:.6f}",
            flush=True,
        )

    print(f"saved outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()