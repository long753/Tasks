"""Fixed FourRoom task execution shared by the WAM evaluation commands."""

from __future__ import annotations

import math
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, Self, cast

import numpy as np
from PIL import Image, ImageDraw

from jepa_wam.policy import WAMPolicy
from stable_worldmodel import World
from stable_worldmodel.fourroom_eval import fixed_task_payload


@dataclass(frozen=True)
class TaskResult:
    task_index: int
    episode_index: int
    environment_seed: int
    policy_seed: int
    success: bool
    steps_to_success: int | None
    elapsed_s: float


def evaluate_task(
    world: World,
    policy: WAMPolicy,
    tasks: dict[str, Any],
    task_index: int,
    *,
    seed: int,
    budget: int,
    callables: list[dict[str, Any]],
    video: ExecutionVideo | None = None,
) -> TaskResult:
    payload = fixed_task_payload(tasks, task_index)
    episode_index = payload['episodes_idx'][0]
    policy_seed = seed + task_index
    policy.generator.manual_seed(policy_seed)
    start = perf_counter()
    metrics = world.evaluate_from_dataset(
        None, None, None,
        goal_offset_steps=int(tasks["goal_offset_steps"]),
        eval_budget=budget,
        callables=callables,
        load_eval_trajs_path=payload,
        on_step=video,
    )
    elapsed = perf_counter() - start
    steps = float(metrics["steps_to_success"][0])
    return TaskResult(
        task_index, episode_index, int(metrics["seeds"][0]), policy_seed,
        bool(metrics["episode_successes"][0]), int(steps) if math.isfinite(steps) else None, elapsed,
    )


class VideoWriter(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> bool | None: ...

    def append_data(self, frame: np.ndarray) -> None: ...


class ExecutionVideo:
    def __init__(self, writer: VideoWriter, task_index: int, budget: int):
        self.writer = writer
        self.task_index = task_index
        self.budget = budget
        self.frames = 0
        self.finished = False
        self.canvas = Image.new("RGB", (512, 288))
        self.draw = ImageDraw.Draw(self.canvas)

    def __call__(self, world: World, step: int) -> None:
        if self.finished:
            return
        frame = world.envs.envs[0].render()
        current = Image.fromarray(np.asarray(frame)).convert("RGB")
        self.canvas.paste(current.resize((256, 256)), (0, 32))
        if step == 0:
            goal = Image.fromarray(world.infos["goal"][0, -1]).convert("RGB")
            self.canvas.paste(goal.resize((256, 256)), (256, 32))
        success = step > 0 and world.terminateds is not None and bool(world.terminateds[0])
        status = "SUCCESS" if success else "BUDGET EXHAUSTED" if step == self.budget else "RUNNING"
        self.draw.rectangle((0, 0, 511, 31), fill="black")
        self.draw.text((8, 2), f"Task {self.task_index} | Step {step} | {status}", fill="white")
        self.draw.text((8, 17), "Actual execution", fill="white")
        self.draw.text((264, 17), "Goal", fill="white")
        self.writer.append_data(np.asarray(self.canvas))
        self.frames += 1
        self.finished = success


@contextmanager
def record_execution(path: Path, fps: int, task_index: int, budget: int) -> Generator[ExecutionVideo]:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with cast(VideoWriter, imageio.get_writer(path, fps=fps, codec="libx264")) as writer:
        yield ExecutionVideo(writer, task_index, budget)
