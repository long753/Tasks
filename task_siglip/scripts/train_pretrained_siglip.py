"""Frozen DINO + BERT-Mini: train only two linear projections and SigLIP logits.

uv run python -m scripts.train_pretrained_siglip
uv run python -m scripts.train_pretrained_siglip --mode evaluate --checkpoint models/<run>/best.pt

This experiment deliberately does not import or modify scripts/train.py.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    BertConfig,
    BertModel,
    ViTConfig,
    ViTModel,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from src.data import ImageCaptionDataset
from src.evaluation.feature_diagnostics import feature_statistics, select_samples
from src.evaluation.retrieval import collect_captions, compute_recall
from src.evaluation.zero_shot import PROMPT_TEMPLATE
from src.model.loss import siglip_cross_entropy_loss

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VISION_NAME = "facebook/dino-vits16"
TEXT_NAME = "google/bert_uncased_L-4_H-256_A-4"
RUN_NAME = "siglip-dino-bert-mini-frozen-coco"


class FrozenSigLIP(nn.Module):
    def __init__(self, vision, text, tokenizer, seed=42):
        super().__init__()
        self.vision = vision.requires_grad_(False)
        self.text = text.requires_grad_(False)
        self.tokenizer = tokenizer
        self.max_length = 64
        # Seed the heads independently of backbone loading/initialization.
        torch.manual_seed(seed)
        self.image_projection = nn.Linear(vision.config.hidden_size, 256)
        self.text_projection = nn.Linear(text.config.hidden_size, 256)
        for projection in (self.image_projection, self.text_projection):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        self.text.eval()
        return self

    def image_backbone_features(self, pixels):
        with torch.no_grad():
            return self.vision(pixel_values=pixels).last_hidden_state[:, 0]

    def text_backbone_features(self, texts, device):
        tokens = self.tokenizer(
            texts, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            hidden = self.text(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1)

    def image_features(self, pixels, *, normalize=True):
        features = self.image_projection(self.image_backbone_features(pixels)).float()
        return F.normalize(features, dim=-1) if normalize else features

    def text_features(self, texts, device, *, normalize=True):
        features = self.text_projection(self.text_backbone_features(texts, device)).float()
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, texts, pixels):
        image = self.image_features(pixels)
        text = self.text_features(texts, pixels.device)
        return self.logit_scale.exp() * (image @ text.T).float() + self.logit_bias


class ImageTextBatch:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        try:
            pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        finally:
            for image in images:
                image.close()
        return {"pixel_values": pixels, "texts": [item["text"] for item in batch]}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def create_model(seed, checkpoint=None, *, load_weights=True):
    """Reconstruct a saved model; load_weights=False gives a random-config reference."""
    vision_name = VISION_NAME if checkpoint is None else checkpoint["pretrained_sources"]["vision"]
    text_name = TEXT_NAME if checkpoint is None else checkpoint["pretrained_sources"]["text"]
    tokenizer = AutoTokenizer.from_pretrained(text_name)
    processor = AutoImageProcessor.from_pretrained(vision_name, use_fast=False)
    if checkpoint is None:
        vision = ViTModel.from_pretrained(vision_name, add_pooling_layer=False)
        text = BertModel.from_pretrained(text_name, add_pooling_layer=False)
    else:
        if not load_weights:
            torch.manual_seed(seed)
        config = checkpoint["encoder_configs"]
        # Reconstruct from config, not from downloaded pretrained weights.
        vision = ViTModel(ViTConfig(**config["vision"]), add_pooling_layer=False)
        text = BertModel(BertConfig(**config["text"]), add_pooling_layer=False)
    model = FrozenSigLIP(vision, text, tokenizer, seed)
    if checkpoint is not None and load_weights:
        model.load_state_dict(checkpoint["model_state_dict"])
    return model, processor


def parameter_counts(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def encoder_fingerprint(model):
    digest = hashlib.sha256()
    for encoder in (model.vision, model.text):
        for name, tensor in encoder.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_state(model, optimizer, scheduler, args, epoch, best_loss):
    return {
        "model_backend": "frozen_dino_bert",
        "model_state_dict": model.state_dict(),
        "encoder_configs": {"vision": model.vision.config.to_dict(), "text": model.text.config.to_dict()},
        "pretrained_sources": {"vision": VISION_NAME, "text": TEXT_NAME},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_loss,
        "args": vars(args),
    }


def run_epoch(model, loader, device, optimizer=None, scheduler=None, max_steps=None, ready_path=None, epoch=1):
    training = optimizer is not None
    model.train(training)
    total_loss, image_count = 0.0, 0
    durations = []
    started = previous = time.perf_counter()
    for step, batch in enumerate(tqdm(loader, desc="Train" if training else "Val", unit="batch"), start=1):
        pixels = batch["pixel_values"].to(device)
        size = len(pixels)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
        ):
            logits = model(batch["texts"], pixels)
            loss = siglip_cross_entropy_loss(logits, torch.eye(size, dtype=torch.bool, device=device))
        value = loss.detach().float().item()
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite SigLIP loss")
        if training:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        total_loss += value * size
        image_count += size
        if device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        durations.append(now - previous)
        previous = now
        if ready_path is not None and step == 64:
            readiness = {
                "status": "training",
                "epoch": epoch,
                "completed_batches": step,
                "batches_per_epoch": len(loader),
                "loss": value,
                "steady_step_seconds": statistics.mean(durations[4:]),
                "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda"
                else None,
            }
            write_json(ready_path, readiness)
            print("FORMAL_TRAINING_READY " + json.dumps(readiness), flush=True)
        if max_steps is not None and step >= max_steps:
            break
    return {
        "loss": total_loss / image_count,
        "seconds": time.perf_counter() - started,
        "steps": len(durations),
        "num_images": image_count,
        "steady_step_seconds": statistics.mean(durations[4:] or durations),
    }


@torch.inference_mode()
def evaluate_model(model, processor, args, device, label):
    model.eval()
    result = {
        "label": label,
        "pretrained_sources": {"vision": VISION_NAME, "text": TEXT_NAME},
        "evaluation_precision": "float32",
        "retrieval": {},
    }
    collate = ImageTextBatch(processor)
    for name, manifest in (("coco", "coco_val2017.jsonl"), ("flickr8k", "flickr8k_test.jsonl")):
        dataset = ImageCaptionDataset(
            PROJECT_ROOT / args.data_root / "manifests" / manifest, root=PROJECT_ROOT, random_caption=False
        )
        texts, owners, _, captions_by_image = collect_captions(dataset)
        image_vectors, text_vectors = [], []
        for start in tqdm(range(0, len(dataset), args.batch_size), desc=f"{label} {name} images"):
            batch = collate([dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))])
            image_vectors.append(model.image_features(batch["pixel_values"].to(device)).cpu())
        for start in tqdm(range(0, len(texts), args.batch_size), desc=f"{label} {name} texts"):
            text_vectors.append(model.text_features(texts[start : start + args.batch_size], device).cpu())
        image_vectors, text_vectors = torch.cat(image_vectors), torch.cat(text_vectors)
        result["retrieval"][name] = {
            "num_images": len(dataset),
            "num_texts": len(texts),
            "metrics": compute_recall(image_vectors.to(device) @ text_vectors.to(device).T, owners, ks=(1,)),
        }
        if name == "coco":
            samples = select_samples(dataset, 1024, args.seed)
            image_indices = [sample["dataset_index"] for sample in samples]
            text_indices = [captions_by_image[sample["dataset_index"]][sample["caption_index"]] for sample in samples]
            result["features"] = {
                "image": feature_statistics(image_vectors[image_indices])[0],
                "text": feature_statistics(text_vectors[text_indices])[0],
            }
            result["sampling"] = {"seed": args.seed, "num_samples": len(samples), "samples": samples}
    dataset = CIFAR100(PROJECT_ROOT / args.data_root / "cifar100", train=False, download=False)
    prompts = [PROMPT_TEMPLATE.format(name.replace("_", " ")) for name in dataset.classes]
    class_vectors = torch.cat(
        [model.text_features(prompts[i : i + args.batch_size], device) for i in range(0, len(prompts), args.batch_size)]
    )
    correct = 0
    for start in tqdm(range(0, len(dataset), args.batch_size), desc=f"{label} CIFAR100"):
        batch = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
        images, labels = zip(*batch)
        try:
            pixels = processor(images=list(images), return_tensors="pt")["pixel_values"].to(device)
        finally:
            for image in images:
                image.close()
        predicted = (model.image_features(pixels) @ class_vectors.T).argmax(dim=1)
        correct += (predicted == torch.tensor(labels, device=device)).sum().item()
    result["cifar100"] = {
        "num_images": len(dataset),
        "Top1": 100 * correct / len(dataset),
        "prompt_template": PROMPT_TEMPLATE,
    }
    write_json(PROJECT_ROOT / args.log_dir / "evaluation" / f"{label}.json", result)
    return result


def write_evaluation_comparison(log_dir):
    reports = [
        json.loads(path.read_text())
        for label in ("before", "best", "last")
        if (path := log_dir / "evaluation" / f"{label}.json").exists()
    ]
    lines = [
        "# 冻结 DINO + BERT-Mini：投影训练前后对比",
        "",
        "这里只训练投影和 logit 参数；before 的投影是随机初始化，编码器已经预训练。",
        "",
        "| 状态 | COCO 图→文 R@1 | COCO 文→图 R@1 | Flickr8k 图→文 R@1 | Flickr8k 文→图 R@1 | CIFAR100 Top1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        values = [
            report["retrieval"][name]["metrics"][direction]["R@1"]
            for name in ("coco", "flickr8k")
            for direction in ("image_to_text", "text_to_image")
        ]
        values.append(report["cifar100"]["Top1"])
        lines.append(f"| {report['label']} | " + " | ".join(f"{value:.2f}" for value in values) + " |")
    lines += [
        "",
        "## 投影后共同空间（256维）的特征诊断",
        "",
        "| 状态 | 图像有效秩 | 图像总方差 | 文本有效秩 | 文本总方差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for report in reports:
        values = [
            report["features"][modality][key]
            for modality in ("image", "text")
            for key in ("effective_rank", "total_variance")
        ]
        lines.append(
            f"| {report['label']} | "
            + " | ".join("N/A" if value is None else f"{value:.4f}" for value in values)
            + " |"
        )
    lines += ["", "有效秩和总方差不是性能分数；冻结编码器本身的特征不变，变化来自投影。", ""]
    (log_dir / "evaluation" / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def plot_losses(history, path):
    figure, axis = plt.subplots()
    for key in ("train_loss", "val_loss"):
        axis.plot([row["epoch"] for row in history], [row[key] for row in history], label=key)
    axis.set(xlabel="Epoch", ylabel="SigLIP loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def smoke(model, processor, train_loader, val_loader, optimizer, scheduler, args, device):
    before_hash = encoder_fingerprint(model)
    before_heads = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    sample = next(iter(train_loader))
    pixels = sample["pixel_values"].to(device)
    model.train()
    assert not model.vision.training and not model.text.training
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        fixed_before = model.text_backbone_features(sample["texts"], device).float().cpu()
        fixed_again = model.text_backbone_features(sample["texts"], device).float().cpu()
    assert torch.equal(fixed_before, fixed_again), "Frozen BERT dropout must be disabled"
    train_stats = run_epoch(model, train_loader, device, optimizer, scheduler, max_steps=64)
    val_stats = run_epoch(model, val_loader, device)
    assert train_stats["steps"] == 64 and val_stats["num_images"] == 5000
    assert encoder_fingerprint(model) == before_hash
    assert all(p.grad is None for encoder in (model.vision, model.text) for p in encoder.parameters())
    changed = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and not torch.equal(before_heads[name], p.detach().cpu())
    ]
    assert any(name.startswith("image_projection") for name in changed)
    assert any(name.startswith("text_projection") for name in changed)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad
    )
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        fixed_after = model.text_backbone_features(sample["texts"], device).float().cpu()
        expected_logits = model(sample["texts"], pixels).float().cpu()
    assert torch.equal(fixed_before, fixed_after)
    state = checkpoint_state(model, optimizer, scheduler, args, 1, val_stats["loss"])
    path = PROJECT_ROOT / args.model_dir / "smoke.pt"
    start = time.perf_counter()
    torch.save(state, path)
    save_seconds = time.perf_counter() - start
    saved = torch.load(path, map_location="cpu", weights_only=True)
    restored, _ = create_model(args.seed, saved)
    restored = restored.to(device).eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        actual_logits = restored(sample["texts"], pixels).float().cpu()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    epoch_seconds = train_stats["steady_step_seconds"] * len(train_loader) + val_stats["seconds"] + 2 * save_seconds
    report = {
        "status": "passed",
        "parameter_counts": parameter_counts(model),
        "batch_size": args.batch_size,
        "epochs": args.num_epochs,
        "training": train_stats,
        "validation": val_stats,
        "encoder_fingerprint_unchanged": True,
        "encoder_gradients_absent": True,
        "frozen_text_features_unchanged": True,
        "changed_trainable_parameters": changed,
        "checkpoint_logits_identical": True,
        "checkpoint_save_seconds": save_seconds,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None,
        "steps_per_epoch": len(train_loader),
        "estimated_epoch_seconds": epoch_seconds,
        "estimated_training_hours": epoch_seconds * args.num_epochs / 3600,
        "timing_note": "Includes validation and two checkpoint writes per epoch; excludes before/best/last benchmark evaluation.",
    }
    write_json(PROJECT_ROOT / args.log_dir / "smoke_report.json", report)
    print("FROZEN_PREFLIGHT_PASSED " + json.dumps(report), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "smoke", "evaluate"), default="train")
    parser.add_argument("--checkpoint", help="Required only for evaluate mode")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--model_dir", default=f"models/{RUN_NAME}")
    parser.add_argument("--log_dir", default=f"outputs/{RUN_NAME}")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.batch_size < 2 or args.num_epochs < 1 or args.num_workers < 0:
        parser.error("batch_size >= 2, num_epochs >= 1 and num_workers >= 0 are required")
    if args.mode == "evaluate" and args.checkpoint is None:
        parser.error("--checkpoint is required for evaluate mode")
    if args.mode != "evaluate" and args.checkpoint is not None:
        parser.error("--checkpoint is only supported for evaluate mode; training starts from pretrained encoders")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment requires BF16 support on CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_dir, log_dir = PROJECT_ROOT / args.model_dir, PROJECT_ROOT / args.log_dir
    if args.mode != "evaluate" and model_dir.exists() and any(model_dir.glob("*.pt")):
        raise FileExistsError(f"Refusing to overwrite existing model checkpoints: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        torch.load(PROJECT_ROOT / args.checkpoint, map_location="cpu", weights_only=True)
        if args.mode == "evaluate"
        else None
    )
    if checkpoint is not None:
        args.seed = checkpoint["args"]["seed"]
    model, processor = create_model(args.seed, checkpoint)
    model = model.to(device)
    print(
        "MODEL_READY "
        + json.dumps(
            {
                "sources": [VISION_NAME, TEXT_NAME],
                "parameters": parameter_counts(model),
                "device": str(device),
                "precision": "bf16" if device.type == "cuda" else "fp32",
                "frozen_encoders": True,
            }
        ),
        flush=True,
    )
    if args.mode == "evaluate":
        evaluate_model(model, processor, args, device, Path(args.checkpoint).stem)
        write_evaluation_comparison(log_dir)
        return
    collate = ImageTextBatch(processor)
    train_data = ImageCaptionDataset(
        PROJECT_ROOT / args.data_root / "manifests/coco_train2017.jsonl", root=PROJECT_ROOT
    )
    val_data = ImageCaptionDataset(
        PROJECT_ROOT / args.data_root / "manifests/coco_val2017.jsonl", root=PROJECT_ROOT, random_caption=False
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": args.weight_decay},
            {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
    )
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps, min_lr=1e-6
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if args.mode == "smoke":
        smoke(model, processor, train_loader, val_loader, optimizer, scheduler, args, device)
        return
    write_json(
        log_dir / "run_config.json",
        {
            "args": vars(args),
            "parameter_counts": parameter_counts(model),
            "sources": {"vision": VISION_NAME, "text": TEXT_NAME},
            "encoder_configs": {"vision": model.vision.config.to_dict(), "text": model.text.config.to_dict()},
        },
    )
    evaluate_model(model, processor, args, device, "before")
    write_evaluation_comparison(log_dir)
    print("BASELINE_EVALUATION_COMPLETE; starting one frozen-encoder training run", flush=True)
    writer = SummaryWriter(log_dir=log_dir / "tensorboard")
    best_loss, history = float("inf"), []
    try:
        for epoch in range(1, args.num_epochs + 1):
            print(f"Epoch {epoch}/{args.num_epochs}", flush=True)
            train_stats = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                scheduler,
                ready_path=log_dir / "training_ready.json" if epoch == 1 else None,
                epoch=epoch,
            )
            val_stats = run_epoch(model, val_loader, device)
            improved = val_stats["loss"] < best_loss
            best_loss = min(best_loss, val_stats["loss"])
            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "lr": optimizer.param_groups[0]["lr"],
                "train_seconds": train_stats["seconds"],
                "val_seconds": val_stats["seconds"],
            }
            history.append(row)
            with (log_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            writer.add_scalar("loss/train", row["train_loss"], epoch)
            writer.add_scalar("loss/val", row["val_loss"], epoch)
            writer.add_scalar("learning_rate", row["lr"], epoch)
            writer.flush()
            saved = checkpoint_state(model, optimizer, scheduler, args, epoch, best_loss)
            torch.save(saved, model_dir / "last.pt")
            if improved:
                torch.save(saved, model_dir / "best.pt")
            if epoch % 10 == 0:
                torch.save(saved, model_dir / f"epoch_{epoch:03d}.pt")
            print(json.dumps(row), flush=True)
    finally:
        writer.close()
    plot_losses(history, log_dir / "loss_curve.png")
    for label in ("best", "last"):
        saved = torch.load(model_dir / f"{label}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state_dict"])
        evaluate_model(model, processor, args, device, label)
    write_evaluation_comparison(log_dir)
    print(f"TRAINING_AND_EVALUATION_COMPLETE {log_dir / 'evaluation/comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
