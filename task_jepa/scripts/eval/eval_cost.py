"""Benchmark FourRoom policies in closed loop and on identical recorded observations."""

import argparse
import csv
import os
from contextlib import ExitStack, nullcontext, redirect_stdout
from dataclasses import asdict, fields
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from h_jepa.planning_eval import (
    _build_eval_process_from_policy_normalizer,
    _build_policy,
    img_transform,
    load_eval_config,
)
from jepa_wam.policy import WAMPolicy
from stable_worldmodel.fourroom_eval import (
    ClosedLoopResult,
    CountingPolicy,
    PlanningPolicy,
    ReplayResult,
    closed_loop,
    load_fourroom_protocol,
    replay_observations,
    time_replay,
)
from stable_worldmodel.paths import add_storage_arguments, resolve_path, resolve_run_name
from stable_worldmodel.policy import WorldModelPolicy
from stable_worldmodel.world import World


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_storage_arguments(parser, include_models=False)
    parser.add_argument('--model', action='append', required=True, nargs=3, metavar=('NAME', 'CONFIG', 'CHECKPOINT'),
                        help='Repeat to benchmark more models. CONFIG is wam or a FourRoom eval config name.')
    parser.add_argument('--mode', choices=('closed-loop', 'replay', 'both'), default='both')
    parser.add_argument('--run-name')
    parser.add_argument('--device', default='cuda', type=torch.device)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--closed-tasks', type=int, default=50, help='Use the first N fixed tasks for closed loop')
    parser.add_argument('--replay-tasks', type=int, default=10, help='Use N evenly spaced fixed trajectories for replay')
    args = parser.parse_args()
    if args.device.type != 'cuda':
        parser.error('--device must be cuda or cuda:N')
    if len({row[0] for row in args.model}) != len(args.model):
        parser.error('Each --model must have a unique result name')
    for row in args.model:
        _, config, checkpoint = row
        if config != 'wam' and (not config.startswith('fourroom_') or Path(config).name != config):
            parser.error(f'Not a FourRoom configuration: {config}')
        path = resolve_path(checkpoint)
        if not path.is_file():
            parser.error(f'Checkpoint does not exist: {path}')
        row[2] = str(path)
    return args


def load_policy(
    config: str, checkpoint: str, args: argparse.Namespace, budget: int,
) -> tuple[PlanningPolicy, tuple[torch.Generator, ...]]:
    if config == 'wam':
        policy = WAMPolicy.from_checkpoint(Path(checkpoint), device=str(args.device), seed=args.seed)
        return policy, (policy.generator,)

    cfg = load_eval_config(config, datasets=args.datasets)
    cfg.checkpoint = checkpoint
    cfg.seed = args.seed
    cfg.eval.eval_budget = budget
    transform = img_transform(cfg)
    jepa = cast(WorldModelPolicy, _build_policy(
        cfg, _build_eval_process_from_policy_normalizer(cfg), {'pixels': transform, 'goal': transform},
    ))
    solver = jepa.solver
    levels = getattr(solver, 'level_solvers', {1: solver})
    return jepa, tuple(level.torch_gen for level in levels.values())


def benchmark_model(
    spec: list[str], args: argparse.Namespace, protocol: DictConfig, tasks: dict[str, Any],
    observations: dict[int, list[dict[str, np.ndarray]]],
    closed_writer: csv.DictWriter | None, replay_writer: csv.DictWriter | None,
) -> dict[str, str | int | float | None]:
    name, config, checkpoint = spec
    budget = int(protocol.eval.eval_budget)
    callables = cast(list[dict[str, Any]], OmegaConf.to_object(protocol.eval.callables))
    world_options = cast(dict[str, Any], OmegaConf.to_object(protocol.world))
    image_size = int(protocol.eval.img_size)
    inner, generators = load_policy(config, checkpoint, args, budget)
    policy = CountingPolicy(inner, partial(torch.cuda.synchronize, args.device))
    warmup = next(iter(observations.values()))[0] if observations else replay_observations(tasks, 0)[0]
    world = World(**world_options, image_shape=(image_size, image_size))
    closed: list[ClosedLoopResult] = []
    latencies: dict[str, list[float]] = {kind: [] for kind in ('first_plan', 'replan', 'cached_action')}
    try:
        world.set_policy(policy)
        # Preserve the shipped H-JEPA gradient-solver attention backend.
        with nullcontext() if config == 'wam' else sdpa_kernel(SDPBackend.MATH):
            policy.reset()
            policy.get_action(warmup.copy())
            if closed_writer is not None:
                for index in tqdm(range(args.closed_tasks), desc=f'{name} closed loop'):
                    for generator in generators:
                        generator.manual_seed(args.seed + index)
                    row = closed_loop(
                        world, policy, tasks, index, model=name, budget=budget,
                        policy_seed=args.seed + index, callables=callables,
                    )
                    closed_writer.writerow(asdict(row))
                    closed.append(row)
            if replay_writer is not None:
                for index, clip in tqdm(observations.items(), desc=f'{name} replay'):
                    for generator in generators:
                        generator.manual_seed(args.seed + index)
                    for sample in time_replay(policy, clip, model=name, task_index=index):
                        replay_writer.writerow(asdict(sample))
                        latencies[sample.kind].append(sample.elapsed_ms)
    finally:
        world.close()

    summary: dict[str, str | int | float | None] = {'model': name, 'config': config, 'checkpoint': checkpoint}
    if closed:
        times = [row.elapsed_s for row in closed]
        steps = sum(row.env_steps for row in closed)
        inference_s = sum(row.inference_s for row in closed)
        success_steps = [row.steps_to_success for row in closed if row.steps_to_success is not None]
        summary.update(
            closed_tasks=len(closed), closed_env_steps=steps, closed_successes=len(success_steps),
            success_rate_pct=100 * len(success_steps) / len(closed),
            closed_mean_steps_to_success=float(np.mean(success_steps)) if success_steps else None,
            closed_inference_s=inference_s, closed_inference_ms_per_step=1000 * inference_s / steps,
            closed_total_s=sum(times), closed_task_p50_s=float(np.median(times)),
            closed_task_p95_s=float(np.percentile(times, 95)), closed_planning_calls=sum(row.planning_calls for row in closed),
        )
    if observations:
        for kind, samples in latencies.items():
            summary[f'{kind}_count'] = len(samples)
            summary[f'{kind}_p50_ms'] = float(np.median(samples)) if samples else None
            summary[f'{kind}_p95_ms'] = float(np.percentile(samples, 95)) if samples else None
    return summary


def main() -> None:
    args = parse_args()
    protocol = load_fourroom_protocol(resolve_path(args.datasets))
    task_path = Path(protocol.load_eval_trajs_path)
    tasks = torch.load(task_path, map_location='cpu', weights_only=False)
    num_tasks = int(protocol.eval.num_eval)
    if len(tasks['data']) != num_tasks:
        raise ValueError(f'Expected {num_tasks} fixed tasks in {task_path}, found {len(tasks["data"])}')
    if not 1 <= args.closed_tasks <= num_tasks or not 1 <= args.replay_tasks <= num_tasks:
        raise ValueError(f'Task counts must be between 1 and {num_tasks}')
    indices = np.linspace(0, num_tasks - 1, args.replay_tasks, dtype=int).tolist() if args.mode != 'closed-loop' else []
    observations = {index: replay_observations(tasks, index) for index in indices}
    run_name = resolve_run_name(args.run_name, prefix='fourroom_cost')
    output = resolve_path(args.outputs) / run_name

    with torch.cuda.device(args.device), ExitStack() as stack:
        output.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(OmegaConf.create({
            **vars(args), 'device': str(args.device), 'run_name': run_name, 'task_path': str(task_path),
            'eval_budget': int(protocol.eval.eval_budget), 'goal_state_source': 'goal_xy',
            'closed_stop_on_success': True,
            'replay_task_indices': indices, 'gpu': torch.cuda.get_device_name(args.device),
            'torch_version': str(torch.__version__), 'jepa_sdpa': 'math',
        }), output / 'eval_config.yaml')
        closed_writer = replay_writer = None
        if args.mode != 'replay':
            stream = stack.enter_context((output / 'closed_loop.csv').open('w', newline='', buffering=1))
            closed_writer = csv.DictWriter(stream, fieldnames=[f.name for f in fields(ClosedLoopResult)])
            closed_writer.writeheader()
        if observations:
            stream = stack.enter_context((output / 'replay.csv').open('w', newline='', buffering=1))
            replay_writer = csv.DictWriter(stream, fieldnames=[f.name for f in fields(ReplayResult)])
            replay_writer.writeheader()
        stream = stack.enter_context((output / 'summary.csv').open('w', newline='', buffering=1))
        quiet = stack.enter_context(Path(os.devnull).open('w'))
        writer = None
        for spec in args.model:
            with redirect_stdout(quiet):
                summary = benchmark_model(spec, args, protocol, tasks, observations, closed_writer, replay_writer)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(summary))
                writer.writeheader()
            writer.writerow(summary)
            torch.cuda.empty_cache()
            print(f'{spec[0]} finished: {output / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
