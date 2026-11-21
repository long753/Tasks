"""Train a task-conditioned WAM on FourRoom HDF5 trajectories."""

import argparse
import csv
import json
import math
import os
from contextlib import ExitStack
from itertools import islice

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from jepa_wam.data import GoalClips
from jepa_wam.model import WAM
from stable_worldmodel.paths import add_storage_arguments, resolve_path, resolve_run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_storage_arguments(parser)
    parser.add_argument("--encoder-weights", required=True, help="LeWM ViT state_dict, not a full model object")
    parser.add_argument("--run-name")
    parser.add_argument("--train-dataset", default="fourroom_tp35_d1", help="HDF5 name without .h5")
    parser.add_argument("--val-dataset", default="fourroom_tp35_d1_val", help="HDF5 name without .h5")
    parser.add_argument("--horizon", type=int, default=5, help="Short action chunk in raw environment steps")
    parser.add_argument("--goal-max", type=int, default=75, help="Farthest relabeled goal in raw environment steps")
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument(
        "--teacher-fraction", type=float, default=0.7, help="Fraction of optimizer steps with EMA subgoals"
    )
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per process (per GPU)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, help="Cap total updates for a smoke test")
    parser.add_argument("--val-batches", type=int, help="Cap validation batches per epoch for a smoke test")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if not 0 <= args.teacher_fraction <= 1 or not 0 <= args.ema_decay < 1:
        parser.error("Expected teacher fraction in [0, 1] and EMA decay in [0, 1)")
    if not 1 <= args.horizon <= args.goal_max:
        parser.error("Expected 1 <= horizon <= goal-max")
    for name in ("epochs", "batch_size", "log_every", "max_steps", "val_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    try:
        train_wam(args, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def train_wam(args: argparse.Namespace, device: torch.device) -> None:
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    torch.manual_seed(args.seed)
    use_bf16 = device.type == "cuda"
    root = resolve_path(args.datasets)
    train_data = GoalClips(
        root / f"{args.train_dataset}.h5", args.horizon, args.goal_max, use_proprio=not args.image_only, training=True
    )
    val_data = GoalClips(
        root / f"{args.val_dataset}.h5", args.horizon, args.goal_max, use_proprio=not args.image_only, training=False
    )
    train_sampler = DistributedSampler(train_data, seed=args.seed, drop_last=True) if distributed else None
    val_sampler = DistributedSampler(val_data, shuffle=False) if distributed else None
    train = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        pin_memory=use_bf16,
    )
    val = DataLoader(
        val_data, batch_size=args.batch_size, num_workers=args.workers, sampler=val_sampler, pin_memory=use_bf16
    )
    total_steps = len(train) * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    if total_steps == 0 or len(val) == 0:
        raise ValueError("Training needs a full batch and validation needs at least one clip")
    teacher_steps = math.ceil(total_steps * args.teacher_fraction)

    model_config = {"horizon": args.horizon, "action_dim": train_data.action_dim, "state_dim": train_data.state_dim}
    model = WAM(**model_config)
    weights = torch.load(resolve_path(args.encoder_weights), map_location="cpu", weights_only=True)
    model.initialize_encoder(weights, train_data.proprio_stats() if train_data.state_dim else None)
    model.to(device)
    training_model = (
        DistributedDataParallel(model, device_ids=[device.index] if use_bf16 else None, broadcast_buffers=False)
        if distributed
        else model
    )
    # Start both encoders identically but draw distinct clips and Flow Matching noise on each rank.
    torch.manual_seed(args.seed + rank)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    run_names = [resolve_run_name(args.run_name, prefix="fourroom_wam") if rank == 0 else ""]
    if distributed:
        dist.broadcast_object_list(run_names, src=0)
    run_name = run_names[0]
    model_dir = resolve_path(args.models) / run_name
    output_dir = resolve_path(args.outputs) / run_name
    if rank == 0:
        model_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)
        config = {
            **vars(args),
            "run_name": run_name,
            "total_steps": total_steps,
            "teacher_steps": teacher_steps,
            "world_size": world_size,
            "global_batch_size": args.batch_size * world_size,
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(
            f"{run_name}: {total_steps} updates, {teacher_steps} EMA then {total_steps - teacher_steps} predicted; "
            f"global batch {config['global_batch_size']}",
            flush=True,
        )

    def compute(batch: dict[str, Tensor], predicted: bool) -> tuple[Tensor, dict[str, Tensor]]:
        batch = {key: value.to(device, non_blocking=use_bf16) for key, value in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            losses = training_model(batch, use_prediction=predicted)
            total = losses["world"] + args.action_weight * losses["action"] + args.sigreg_weight * losses["sigreg"]
        return total, {"loss": total, **losses}

    step = 0
    fields = ("split", "epoch", "step", "source", "loss", "world", "action", "sigreg", "subgoal_rmse")
    path = model_dir / "last.pt"
    with ExitStack() as stack:
        log = None
        if rank == 0:
            stream = stack.enter_context((output_dir / "metrics.csv").open("w", newline="", buffering=1))
            log = csv.DictWriter(stream, fieldnames=fields)
            log.writeheader()
        for epoch in range(1, args.epochs + 1):
            training_model.train()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch - 1)
            count = min(len(train), total_steps - step)
            for batch in tqdm(islice(train, count), total=count, desc=f"Epoch {epoch}", disable=rank != 0):
                predicted = step >= teacher_steps
                source = "predicted" if predicted else "ema"
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = compute(batch, predicted)
                loss.backward()
                optimizer.step()
                model.update_target(args.ema_decay)
                step += 1
                if step in (1, teacher_steps, teacher_steps + 1, total_steps) or step % args.log_every == 0:
                    values = torch.stack([value.detach() for value in metrics.values()])
                    if distributed:
                        dist.all_reduce(values)
                        values.div_(world_size)
                    if log is not None:
                        log.writerow(
                            dict(
                                split="train",
                                epoch=epoch,
                                step=step,
                                source=source,
                                **dict(zip(metrics, values.tolist(), strict=True)),
                            )
                        )

            training_model.eval()
            val_sum = torch.zeros(5, device=device)
            samples = 0
            with torch.no_grad():
                for batch in islice(val, args.val_batches):
                    _, metrics = compute(batch, predicted)
                    size = batch["actions"].size(0)
                    val_sum.add_(torch.stack(list(metrics.values())), alpha=size)
                    samples += size
            val_sum.div_(samples)
            if distributed:
                dist.all_reduce(val_sum)
                val_sum.div_(world_size)
            if log is not None:
                val_metrics = dict(zip(fields[4:], val_sum.tolist(), strict=True))
                log.writerow(dict(split="val", epoch=epoch, step=step, source=source, **val_metrics))
                print(f"Validation: loss={val_metrics['loss']:.4f}  action={val_metrics['action']:.4f}", flush=True)
                pending = path.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "model_config": model_config, "step": step}, pending)
                pending.replace(path)
            if step == total_steps:
                break
    if rank == 0:
        print(f"Checkpoint: {path}")


if __name__ == "__main__":
    main()
