"""Evaluate WAM on the 50 fixed FourRoom tasks."""

import argparse
import csv
from contextlib import nullcontext
from dataclasses import asdict, fields
from pathlib import Path
from statistics import mean
from typing import Any, cast

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from jepa_wam.evaluation import TaskResult, evaluate_task, record_execution
from jepa_wam.policy import WAMPolicy
from stable_worldmodel import World
from stable_worldmodel.fourroom_eval import load_fourroom_protocol
from stable_worldmodel.paths import add_storage_arguments, resolve_path, resolve_run_name


def parse_args(*, single_task: bool) -> argparse.Namespace:
    description = "Record one fixed FourRoom task with WAM." if single_task else __doc__
    parser = argparse.ArgumentParser(description=description)
    add_storage_arguments(parser, include_models=False)
    parser.add_argument("--checkpoint", required=True, help="WAM last.pt checkpoint")
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for policy noise")
    parser.add_argument("--ode-steps", type=int, default=8, help="Euler steps for each flow")
    parser.add_argument("--replan-steps", type=int, default=2, help="Environment steps between replans")
    if single_task:
        parser.add_argument("--task-index", type=int, default=0, help="Zero-based index in the fixed task file")
        parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()
    if single_task and (args.task_index < 0 or args.video_fps <= 0):
        parser.error("--task-index must be nonnegative and --video-fps must be positive")
    return args


def main(*, single_task: bool = False) -> None:
    args = parse_args(single_task=single_task)
    protocol = load_fourroom_protocol(resolve_path(args.datasets))
    task_path = Path(protocol.load_eval_trajs_path)
    tasks = torch.load(task_path, map_location="cpu", weights_only=False)
    num_tasks = len(tasks["data"])
    if num_tasks != int(protocol.eval.num_eval):
        raise ValueError(f"Expected {protocol.eval.num_eval} fixed tasks in {task_path}, found {num_tasks}")
    if single_task and args.task_index >= num_tasks:
        raise ValueError(f"--task-index must be in [0, {num_tasks - 1}]")
    indices = (args.task_index,) if single_task else range(num_tasks)

    checkpoint = resolve_path(args.checkpoint)
    policy = WAMPolicy.from_checkpoint(
        checkpoint, device=args.device, seed=args.seed, ode_steps=args.ode_steps, replan_steps=args.replan_steps,
    )
    budget = int(protocol.eval.eval_budget)
    world_options = cast(dict[str, Any], OmegaConf.to_object(protocol.world))
    callables = cast(list[dict[str, Any]], OmegaConf.to_object(protocol.eval.callables))
    image_size = int(protocol.eval.img_size)

    prefix = "fourroom_wam_once" if single_task else "fourroom_wam_eval"
    run_name = resolve_run_name(args.run_name, prefix=prefix)
    output_dir = resolve_path(args.outputs) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create({
        **vars(args), "run_name": run_name, "checkpoint": str(checkpoint), "task_path": str(task_path),
        "task_indices": list(indices), "goal_offset_steps": int(tasks["goal_offset_steps"]), "eval_budget": budget,
        "goal_state_source": "goal_xy", "world": world_options, "callables": callables,
    }), output_dir / "eval_config.yaml")

    rows: list[TaskResult] = []
    video_metrics: dict[str, str | int | bool | None] = {}
    world = World(**world_options, image_shape=(image_size, image_size))
    try:
        world.set_policy(policy)
        with (output_dir / "tasks.csv").open("w", newline="", buffering=1) as stream:
            writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(TaskResult)])
            writer.writeheader()
            for task_index in tqdm(indices, desc="FourRoom tasks", disable=single_task):
                path = output_dir / "videos" / f"task_{task_index:03d}.mp4"
                recording = record_execution(path, args.video_fps, task_index, budget) if single_task else nullcontext()
                with recording as video:
                    result = evaluate_task(
                        world, policy, tasks, task_index, seed=args.seed, budget=budget, callables=callables, video=video,
                    )
                if video is not None:
                    video_metrics = {
                        "task_index": task_index, "video": str(path.relative_to(output_dir)),
                        "success": result.success, "steps_to_success": result.steps_to_success,
                        "video_fps": args.video_fps, "video_frames": video.frames,
                    }
                rows.append(result)
                writer.writerow(asdict(result))
    finally:
        world.close()

    successful_steps = [row.steps_to_success for row in rows if row.steps_to_success is not None]
    successes = sum(row.success for row in rows)
    OmegaConf.save(OmegaConf.create({
        "num_tasks": len(rows), "num_successes": successes, "success_rate": 100 * successes / len(rows),
        "steps_to_success_success_only": mean(successful_steps) if successful_steps else None,
        "evaluation_time": sum(row.elapsed_s for row in rows), **video_metrics,
    }), output_dir / "metrics.yaml")
    print(f"{successes}/{len(rows)} successful; results: {output_dir}")


if __name__ == "__main__":
    main()
