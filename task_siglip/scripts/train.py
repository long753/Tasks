"""A small, single-GPU training script for learning SigLIP.

uv run python scripts/train.py \
    --config configs/tiny.yaml \
    --dataset coco \
    --num_epochs 20 \
    --batch_size 16
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, get_cosine_with_min_lr_schedule_with_warmup

from src.data import ImageCaptionDataset
from src.model.loss import siglip_cross_entropy_loss
from src.model.siglip import SigLIPModel

PROCESSOR_NAME = "google/siglip-base-patch16-224"


class ImageTextCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        pixels = self.processor(images=[item["image"] for item in batch], return_tensors="pt")["pixel_values"]
        return {"texts": [item["text"] for item in batch], "pixel_values": pixels}


def parse_args(argv=None, *, default_config=None, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=default_config is None, default=default_config)
    parser.add_argument("--dataset", choices=("flickr8k", "coco"), default="flickr8k")
    parser.add_argument("--num_epochs", type=int, default=10, help="Total epochs, including resumed epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate before cosine decay")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--output_dir")
    parser.add_argument("--data_root", default="data", help="Data directory containing manifests/")
    parser.add_argument("--log_dir", help="Training logs directory; default: outputs/<model directory name>")
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable CUDA page-locked host memory")
    parser.add_argument("--resume", help="Resume model, optimizer, scaler and scheduler from last.pt")
    parser.add_argument("--train_limit", type=int, default=0, help="0 means all training images")
    parser.add_argument("--val_limit", type=int, default=0, help="0 means all validation images")
    return parser.parse_args(argv)


def load_metrics(metrics_path: Path) -> list[dict[str, float | int]]:
    if not metrics_path.exists():
        return []
    with metrics_path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_metrics(metrics_path: Path, metrics: list[dict[str, float | int]]) -> None:
    with metrics_path.open("w", encoding="utf-8") as stream:
        for metric in metrics:
            stream.write(json.dumps(metric) + "\n")


def plot_loss_curve(metrics: list[dict[str, float | int]], output_path: Path) -> None:
    epochs = [metric["epoch"] for metric in metrics]
    figure, axis = plt.subplots()
    axis.plot(epochs, [metric["train_loss"] for metric in metrics], label="train")
    axis.plot(epochs, [metric["val_loss"] for metric in metrics], label="val")
    axis.set(xlabel="Epoch", ylabel="Loss")
    if metrics:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def train_one_epoch(model, loader, optimizer, scaler: torch.amp.GradScaler, device, dtype, scheduler) -> float:
    model.train()
    total_loss, count = 0.0, 0
    positive_mask = torch.eye(loader.batch_size, dtype=torch.bool, device=device)
    for batch in tqdm(loader, desc="Train", unit="batch"):
        pixels = batch["pixel_values"].to(device)
        batch_size = pixels.shape[0]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits = model(batch["texts"], pixels)
            loss = siglip_cross_entropy_loss(logits, positive_mask[:batch_size, :batch_size])
        # With FP32/BF16 the scaler is disabled: these are ordinary backward/step calls.
        scaler.scale(loss).backward()
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
        total_loss += loss.item() * batch_size
        count += batch_size
    return total_loss / count


@torch.no_grad()
def evaluate(model, loader, device, dtype) -> float:
    model.eval()
    total_loss, count = 0.0, 0
    positive_mask = torch.eye(loader.batch_size, dtype=torch.bool, device=device)
    for batch in tqdm(loader, desc="Val", unit="batch"):
        pixels = batch["pixel_values"].to(device)
        batch_size = pixels.shape[0]
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits = model(batch["texts"], pixels)
            loss = siglip_cross_entropy_loss(logits, positive_mask[:batch_size, :batch_size])
        total_loss += loss.item() * batch_size
        count += batch_size
    return total_loss / count


def run_training(args, *, model_factory=SigLIPModel, model_backend="custom") -> None:
    random.seed(a=args.seed)
    torch.manual_seed(args.seed)

    # 1. Read the model configuration and choose CPU/CUDA precision.
    with (PROJECT_ROOT / args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    precision = args.precision
    if precision == "auto":
        precision = "fp32" if device.type == "cpu" else ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]

    # 2. Load images and captions. The model itself handles tokenization.
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=PROCESSOR_NAME)
    image_size = config["model"]["image_size"]
    processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path=PROCESSOR_NAME,
        use_fast=False,
        size={"height": image_size, "width": image_size},
    )
    train_name, val_name = {
        "flickr8k": ("flickr8k_train.jsonl", "flickr8k_dev.jsonl"),
        "coco": ("coco_train2017.jsonl", "coco_val2017.jsonl"),
    }[args.dataset]
    manifests = PROJECT_ROOT / args.data_root / "manifests"
    train_data = ImageCaptionDataset(manifests / train_name, root=PROJECT_ROOT)
    val_data = ImageCaptionDataset(manifests / val_name, root=PROJECT_ROOT, random_caption=False)
    if args.train_limit:
        train_data = Subset(train_data, range(min(args.train_limit, len(train_data))))
    if args.val_limit:
        val_data = Subset(val_data, range(min(args.val_limit, len(val_data))))
    collator = ImageTextCollator(processor)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda" and not args.no_pin_memory,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda" and not args.no_pin_memory,
    )
    if args.batch_size < 2 or len(train_loader) == 0:
        raise ValueError("Use a batch_size of at least 2 that fits the training subset")

    # 3. Build the random-initialized model, optimizer, and optional FP16 scaler.
    model = model_factory(config=config, tokenizer=tokenizer).to(device)
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        params=[
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
    )
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda" and precision == "fp16")

    # 4. Restore state when requested and keep models separate from run outputs.
    resume_path = PROJECT_ROOT / args.resume if args.resume else None
    default_model_name = (
        f"{config['name']}-{args.dataset}"
        if model_backend == "custom"
        else f"{config['name']}-{model_backend}-{args.dataset}"
    )
    model_dir = (
        PROJECT_ROOT / args.output_dir
        if args.output_dir
        else PROJECT_ROOT / "models" / (resume_path.parent.name if resume_path else default_model_name)
    )
    log_dir = PROJECT_ROOT / args.log_dir if args.log_dir else PROJECT_ROOT / "outputs" / model_dir.name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, best_loss = 0, float("inf")
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_val_loss"]
        del checkpoint

    metrics_path = log_dir / "metrics.jsonl"
    metrics_history = []
    if resume_path is not None:
        metrics_history = [metric for metric in load_metrics(metrics_path) if metric["epoch"] <= start_epoch]
    write_metrics(metrics_path, metrics_history)

    # 5. Train, validate, and save each completed epoch.
    print(f"{config['name']} | {args.dataset} | {device} | {precision} | output={model_dir}")
    writer = SummaryWriter(log_dir=log_dir / "tensorboard", purge_step=start_epoch + 1)
    try:
        for epoch in range(start_epoch + 1, args.num_epochs + 1):
            print(f"Epoch {epoch}/{args.num_epochs}")
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, dtype, scheduler)
            val_loss = evaluate(model, val_loader, device, dtype)
            is_best = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
            metric = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            metrics_history.append(metric)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metric) + "\n")
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.flush()

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_loss,
                "config": config,
                "model_backend": model_backend,
            }
            torch.save(obj=checkpoint, f=model_dir / "last.pt")
            if is_best:
                torch.save(obj=checkpoint, f=model_dir / "best.pt")
            if epoch % 10 == 0:
                torch.save(obj=checkpoint, f=model_dir / f"epoch_{epoch:03d}.pt")
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.3e}")
    finally:
        writer.close()

    plot_loss_curve(metrics_history, output_path=log_dir / "loss_curve.png")


def main(argv=None):
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
