"""统一比较 models 下各模型的 best.pt / last.pt，默认输出到 outputs/compare_models。

uv run python -m scripts.test_model
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import torch

from src.data import ImageCaptionDataset
from src.evaluation import common, feature_diagnostics, retrieval, zero_shot

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOSS_EPOCH_LIMITS = {"siglip-small-coco-50": 49}


def generate_loss_curves(history_root, model_dirs, output_dir):
    """Plot recorded train/val losses once per run; never infer missing epochs."""
    curve_dir = output_dir / "loss_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    curves = []
    for model_dir in model_dirs:
        source = history_root / model_dir.name / "metrics.jsonl"
        with source.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        limit = LOSS_EPOCH_LIMITS.get(model_dir.name)
        if limit is not None:
            records = [row for row in records if row["epoch"] <= limit]
        epochs = [row["epoch"] for row in records]
        if not records or epochs != sorted(set(epochs)):
            raise ValueError(f"Expected nonempty, unique, ascending loss epochs: {source}")
        if not all(math.isfinite(row[key]) for row in records for key in ("train_loss", "val_loss")):
            raise ValueError(f"Nonfinite recorded loss: {source}")
        figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
        for key, label in (("train_loss", "Train"), ("val_loss", "Validation")):
            axis.plot(epochs, [row[key] for row in records], marker=".", markersize=3, label=label)
        axis.set(
            title=f"{model_dir.name} (recorded epochs {epochs[0]}–{epochs[-1]})",
            xlabel="Epoch",
            ylabel="SigLIP loss",
        )
        if len(epochs) > 1:
            axis.set_xlim(epochs[0], epochs[-1])
        axis.legend()
        path = curve_dir / f"{model_dir.name}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        original_path = source.with_name("loss_curve.png")
        original_created = not original_path.exists()
        if original_created:
            shutil.copyfile(path, original_path)
        curves.append(
            {
                "model_dir": model_dir.name,
                "source": str(source),
                "epoch_limit": limit,
                "epochs": epochs,
                "train_loss": [row["train_loss"] for row in records],
                "val_loss": [row["val_loss"] for row in records],
                "plot": path.relative_to(output_dir).as_posix(),
                "source_link": Path(os.path.relpath(source, output_dir)).as_posix(),
                "original_plot": str(original_path),
                "original_plot_created": original_created,
            }
        )
    return curves


@torch.inference_mode()
def evaluate_retrieval(model, processor, corpus, batch_size, device, samples=None):
    """Use the full candidate pool; reuse its features for the fixed diagnostic subset."""
    image_features = retrieval.encode_images(model, corpus["dataset"], processor, batch_size, device)
    text_features = common.encode_texts(model, corpus["texts"], batch_size, device)
    scores = image_features.to(device) @ text_features.to(device).T
    recalls = retrieval.compute_recall(scores, corpus["owners"], ks=(1,))
    metrics = {
        "num_images": len(image_features),
        "num_texts": len(text_features),
        "image_to_text_R1": recalls["image_to_text"]["R@1"],
        "text_to_image_R1": recalls["text_to_image"]["R@1"],
    }
    features = {}
    if samples is not None:
        image_indices = [sample["dataset_index"] for sample in samples]
        text_indices = [
            corpus["captions_by_image"][sample["dataset_index"]][sample["caption_index"]] for sample in samples
        ]
        for modality, vectors in (
            ("image", image_features[image_indices]),
            ("text", text_features[text_indices]),
        ):
            stats, _, _ = feature_diagnostics.feature_statistics(vectors)
            features[modality] = {
                key: stats[key]
                for key in (
                    "status",
                    "dimension",
                    "max_rank",
                    "effective_rank",
                    "effective_rank_ratio",
                    "total_variance",
                    "nonfinite_vectors",
                    "zero_vectors",
                )
            }
    return metrics, features


def write_comparison(output_dir, protocol, results, loss_curves):
    """Keep task metrics separate from per-training loss histories."""
    report = {"protocol": protocol, "results": results, "loss_curves": loss_curves}
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    rows = []
    for result in results:
        row = {"model": result["model_dir"], "weights": result["weights"], "epoch": result["metadata"]["epoch"]}
        for dataset, values in result["retrieval"].items():
            row[f"{dataset}_image_to_text_R1_percent"] = values["image_to_text_R1"]
            row[f"{dataset}_text_to_image_R1_percent"] = values["text_to_image_R1"]
        row["cifar100_Top1_percent"] = result["cifar100_Top1"]
        for modality, values in result["features"].items():
            for key in ("dimension", "effective_rank", "effective_rank_ratio", "total_variance"):
                row[f"{modality}_{key}"] = values[key]
        rows.append(row)
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def number(value, digits=2):
        return "N/A" if value is None else f"{value:.{digits}f}"

    lines = [
        f"# {len(results)} 份模型权重统一对比",
        "",
        "## 测试协议",
        "",
        "- 所有权重使用相同的完整 COCO val2017、Flickr8k test 检索候选池及全部描述；保留同图多描述的正确归属。",
        "- CIFAR100 使用完整测试集和现有统一提示词模板，不训练分类头。",
        f"- 特征统计固定 COCO 验证集的 {protocol['num_samples']} 张不同图片及每图一条描述，seed={protocol['seed']}。",
        "- 有效秩基于中心化单位特征的协方差谱，同时报告比例和总方差；不是越大就一定性能越好。",
        "- 模型训练集、训练轮数和 batch 未对齐：本报告比较实际能力，不能据此单独归因模型规模。",
        "",
        "## 综合性能（百分比，越高越好）",
        "",
        "| 模型 / 权重 | Epoch | COCO 图→文 R@1 | COCO 文→图 R@1 | Flickr8k 图→文 R@1 | Flickr8k 文→图 R@1 | CIFAR100 Top1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        coco, flickr = result["retrieval"]["coco"], result["retrieval"]["flickr8k"]
        values = [
            coco["image_to_text_R1"],
            coco["text_to_image_R1"],
            flickr["image_to_text_R1"],
            flickr["text_to_image_R1"],
            result["cifar100_Top1"],
        ]
        lines.append(
            f"| {result['model_dir']} / {result['weights']} | {result['metadata']['epoch']} | "
            + " | ".join(number(value) for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 特征诊断（不是性能排名）",
            "",
            "| 模型 / 权重 | 图像有效秩 / 维度 | 图像秩比例 | 图像总方差 | 文本有效秩 / 维度 | 文本秩比例 | 文本总方差 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        cells = []
        for modality in ("image", "text"):
            stats = result["features"][modality]
            ratio = stats["effective_rank_ratio"]
            cells.extend(
                [
                    f"{number(stats['effective_rank'])} / {stats['dimension']}",
                    "N/A" if ratio is None else f"{ratio * 100:.2f}%",
                    number(stats["total_variance"], 4),
                ]
            )
        lines.append(f"| {result['model_dir']} / {result['weights']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "秩比例 = 有效秩 / min(样本数−1, 特征维度)。不同规格不能只比较绝对有效秩。",
            "总方差接近零提示不同样本几乎同向；低有效秩需要结合方差和任务性能解释，不能直接认定坍塌。",
            "N/A 表示方向统计无定义，具体异常数量见 summary.json；不会把它替换成正常分数。",
            "",
        ]
    )
    lines.extend(
        [
            "## 训练损失曲线",
            "",
            "每次训练一张图，包含 train/val 两条曲线；best/last 不是两次训练。",
            "曲线只使用 metrics.jsonl 中的真实记录，不平滑、不插值，也不补造缺失轮次。",
            "Small COCO-50 仅使用第1–49轮；其他训练保留各自完整记录。",
            "不同 batch、数据集的绝对 loss 不宜横向排名，因此每次训练分别绘图。",
            "",
            "| 训练 | 实际记录轮次 | 记录数 | 数据来源 | 曲线 |",
            "|---|---:|---:|---|---|",
        ]
    )
    for curve in loss_curves:
        lines.append(
            f"| {curve['model_dir']} | {curve['epochs'][0]}–{curve['epochs'][-1]} | "
            f"{len(curve['epochs'])} | [metrics.jsonl]({curve['source_link']}) | [查看曲线]({curve['plot']}) |"
        )
    lines.append("")
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--output_dir", default="outputs/compare_models")
    parser.add_argument(
        "--history_root", default="outputs", help="Training log directories matching model folder names"
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=1024, help="Fixed COCO subset for feature diagnostics only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--download", action="store_true", help="Allow downloading CIFAR100 if missing")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_samples < 2:
        parser.error("--batch_size must be positive and --num_samples must be at least 2")
    models_dir = PROJECT_ROOT / args.models_dir
    model_dirs = sorted(folder for folder in models_dir.iterdir() if folder.is_dir())
    checkpoints = [folder / filename for folder in model_dirs for filename in ("best.pt", "last.pt")]
    if not checkpoints:
        parser.error(f"No model subdirectories found in {models_dir}")
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            parser.error(f"Missing checkpoint: {checkpoint}")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    data_root = PROJECT_ROOT / args.data_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_curves = generate_loss_curves(PROJECT_ROOT / args.history_root, model_dirs, output_dir)
    corpora = {}
    for name, manifest in (("coco", "coco_val2017.jsonl"), ("flickr8k", "flickr8k_test.jsonl")):
        dataset = ImageCaptionDataset(data_root / "manifests" / manifest, root=PROJECT_ROOT, random_caption=False)
        texts, owners, _, captions_by_image = retrieval.collect_captions(dataset)
        corpora[name] = {"dataset": dataset, "texts": texts, "owners": owners, "captions_by_image": captions_by_image}
    samples = feature_diagnostics.select_samples(corpora["coco"]["dataset"], args.num_samples, args.seed)
    protocol = {
        "device": str(device),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_samples": len(samples),
        "feature_manifest": str(data_root / "manifests" / "coco_val2017.jsonl"),
        "samples": samples,
        "retrieval": {
            name: {"num_images": len(corpus["dataset"]), "num_texts": len(corpus["texts"])}
            for name, corpus in corpora.items()
        },
        "classification": {"dataset": "cifar100", "split": "test", "prompt_template": zero_shot.PROMPT_TEMPLATE},
        "feature_definition": "Centered unit-length retrieval features; covariance entropy effective rank; no logits",
        "loss_epoch_limits": LOSS_EPOCH_LIMITS,
    }
    results = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        print(f"\n[{index}/{len(checkpoints)}] {checkpoint.parent.name}/{checkpoint.name}", flush=True)
        model, processor, metadata = common.load_model(checkpoint, device)
        result = {
            "model_dir": checkpoint.parent.name,
            "weights": checkpoint.stem,
            "metadata": metadata,
            "retrieval": {},
        }
        for name, corpus in corpora.items():
            print(f"Retrieval: {name}", flush=True)
            metrics, features = evaluate_retrieval(
                model, processor, corpus, args.batch_size, device, samples=samples if name == "coco" else None
            )
            result["retrieval"][name] = metrics
            if name == "coco":
                result["features"] = features
        print("Zero-shot: CIFAR100 (10000 test images)", flush=True)
        classification = zero_shot.run_zero_shot(
            model=model,
            processor=processor,
            metadata=metadata,
            dataset_name="cifar100",
            data_root=data_root,
            output_dir=output_dir / "details" / checkpoint.parent.name,
            batch_size=args.batch_size,
            device=device,
            download=args.download,
        )
        result["cifar100_Top1"] = classification["metrics"]["Top1"]
        results.append(result)
        del model, processor, classification
    write_comparison(output_dir, protocol, results, loss_curves)
    print(f"\nCompared {len(results)} checkpoints. Report: {output_dir / 'comparison.md'}")
    return results


if __name__ == "__main__":
    main()
