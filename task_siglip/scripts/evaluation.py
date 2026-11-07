"""Run retrieval, zero-shot classification or feature diagnostics.

Retrieval remains the default task:
uv run python -m scripts.evaluation \
    --checkpoint models/siglip-small-coco-50/last.pt \
    --manifest data/manifests/coco_val2017.jsonl

Zero-shot classification:
uv run python -m scripts.evaluation \
    --task zero-shot \
    --checkpoint models/siglip-small-coco-50/last.pt \
    --dataset cifar100 --data_root data --download

Feature diagnostics:
uv run python -m scripts.evaluation \
  --task feature-diagnostics \
  --checkpoint models/siglip-tiny-official-20/best.pt \
  --manifest data/manifests/coco_val2017.jsonl \
  --num_samples 1024 --batch_size 32 --seed 42
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.evaluation import common, feature_diagnostics, retrieval, zero_shot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=("retrieval", "zero-shot", "feature-diagnostics"), default="retrieval")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Inference batch size, not retrieval candidate-pool size"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output_dir", help="Results directory; default: outputs/<checkpoint directory name>")
    parser.add_argument("--manifest", help="Image-caption manifest; required for retrieval and feature diagnostics")
    parser.add_argument(
        "--num_examples", type=int, default=20, help="Retrieval examples to visualize; 0 writes metrics only"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for sample selection and random initialization")
    parser.add_argument(
        "--num_samples", type=int, default=1024, help="Distinct images to sample for feature diagnostics"
    )
    parser.add_argument("--random_init", action="store_true", help="Diagnose a fresh model using the checkpoint config")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10", help="Zero-shot test dataset")
    parser.add_argument(
        "--data_root", default="data", help="Classification datasets are stored under <data_root>/<dataset>"
    )
    parser.add_argument("--download", action="store_true", help="Allow downloading the classification dataset")
    args = parser.parse_args(argv)
    if args.task in ("retrieval", "feature-diagnostics") and args.manifest is None:
        parser.error(f"--manifest is required for {args.task}")
    if args.task == "retrieval" and args.num_examples < 0:
        parser.error("--num_examples must be non-negative")
    if args.random_init and args.task != "feature-diagnostics":
        parser.error("--random_init is only supported for feature-diagnostics")
    if args.task == "feature-diagnostics" and (args.num_samples < 2 or args.batch_size < 1):
        parser.error("Feature diagnostics require --num_samples >= 2 and --batch_size >= 1")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    output_dir = (
        PROJECT_ROOT / args.output_dir if args.output_dir else PROJECT_ROOT / "outputs" / checkpoint_path.parent.name
    )
    model, processor, metadata = common.load_model(
        checkpoint_path, device, random_init=args.random_init, seed=args.seed
    )

    if args.task == "retrieval":
        manifest_path = PROJECT_ROOT / args.manifest
        report = retrieval.run_retrieval(
            model=model,
            processor=processor,
            metadata=metadata,
            manifest_path=manifest_path,
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
            batch_size=args.batch_size,
            device=device,
            num_examples=args.num_examples,
            seed=args.seed,
        )
        report_path = output_dir / f"{manifest_path.stem}_{checkpoint_path.stem}.json"
    elif args.task == "zero-shot":
        report = zero_shot.run_zero_shot(
            model=model,
            processor=processor,
            metadata=metadata,
            dataset_name=args.dataset,
            data_root=PROJECT_ROOT / args.data_root,
            output_dir=output_dir,
            batch_size=args.batch_size,
            device=device,
            download=args.download,
        )
        report_path = output_dir / "zero_shot" / f"{args.dataset}_{checkpoint_path.stem}.json"
    else:
        report = feature_diagnostics.run_feature_diagnostics(
            model=model,
            processor=processor,
            metadata=metadata,
            manifest_path=PROJECT_ROOT / args.manifest,
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
            batch_size=args.batch_size,
            device=device,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        report_path = Path(report["artifacts"]["report"])

    if args.random_init:
        print(f"Random initialization (seed {args.seed}, {metadata['model']}); config from {args.checkpoint}")
    else:
        print(f"Checkpoint: {args.checkpoint} (epoch {metadata['epoch']}, {metadata['model']})")
    if args.task == "retrieval":
        print(f"Manifest: {args.manifest}")
        print(f"Candidates: {report['num_images']} images, {report['num_texts']} captions")
        for direction, values in report["metrics"].items():
            print(f"{direction}: " + " | ".join(f"{name}={value:.2f}%" for name, value in values.items()))
    elif args.task == "zero-shot":
        print(f"Dataset: {args.dataset} test ({report['num_images']} images, {report['num_classes']} classes)")
        print(" | ".join(f"{name}={value:.2f}%" for name, value in report["metrics"].items()))
        print(f"Saved confusion matrix: {report_path.with_name(report_path.stem + '_confusion.png')}")
    else:
        print(f"Manifest: {args.manifest} | Sampled images: {report['sampling']['num_samples']}")
        for modality in ("image", "text"):
            stats = report["metrics"][modality]
            if stats["status"] == "invalid_directions":
                print(
                    f"{modality}: nonfinite vectors={stats['nonfinite_vectors']}, zero vectors={stats['zero_vectors']}"
                )
            else:
                print(
                    f"{modality}: variance={stats['total_variance']:.6g} | "
                    f"effective rank={stats['effective_rank']:.2f}/{stats['max_rank']} | "
                    f"mean direction norm={stats['mean_vector_norm']:.4f}"
                )
        alignment = report["metrics"]["alignment"]
        if alignment["status"] == "ok":
            print(f"Matched-minus-negative cosine: {alignment['paired_minus_negative_mean']:.6f}")
            for direction, values in alignment["retrieval_percent"].items():
                print(f"{direction}: " + " | ".join(f"{name}={value:.2f}%" for name, value in values.items()))
    print(f"Saved results: {report_path}")
    if args.task == "retrieval" and args.num_examples:
        print(f"Saved visualizations: {output_dir / 'visualizations' / report_path.stem / 'index.html'}")
    return report["metrics"]


if __name__ == "__main__":
    main()
