"""Single-task simulation evaluation with an actual execution/goal video."""

from typing import Protocol, Self, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

from h_jepa.planning_eval import resolve_results_dir, run_planning_eval
from h_jepa.runtime import storage_config
from stable_worldmodel.paths import CONFIG_ROOT
from stable_worldmodel.world import World


class _VideoWriter(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> bool | None: ...

    def append_data(self, frame: np.ndarray) -> None: ...


class _ExecutionVideo:
    """Stream frames without retaining an episode-sized image buffer."""

    def __init__(self, writer: _VideoWriter, task_index: int, eval_budget: int):
        self.writer = writer
        self.task_index = task_index
        self.eval_budget = eval_budget
        self.frames = 0
        self.finished = False
        self.canvas = Image.new("RGB", (512, 288))
        self.draw = ImageDraw.Draw(self.canvas)

    def __call__(self, world: World, step: int) -> None:
        if self.finished:
            return
        # Render the restored live state, not the dataset pixels placed in infos.
        frame = world.envs.envs[0].render()
        if frame is None:
            raise RuntimeError("eval_once requires an environment with RGB-array rendering")
        current = Image.fromarray(np.asarray(frame)).convert("RGB")
        self.canvas.paste(current.resize((256, 256)), (0, 32))
        if step == 0:
            goal = Image.fromarray(np.asarray(world.infos["goal"][0, -1])).convert("RGB")
            self.canvas.paste(goal.resize((256, 256)), (256, 32))
        successful = step > 0 and world.terminateds is not None and bool(world.terminateds[0])
        status = "SUCCESS" if successful else "BUDGET EXHAUSTED" if step >= self.eval_budget else "RUNNING"
        self.draw.rectangle((0, 0, 511, 31), fill="black")
        self.draw.text((8, 2), f"Task {self.task_index} | step {step} | {status}", fill="white")
        self.draw.text((8, 17), "Actual execution", fill="white")
        self.draw.text((264, 17), "Goal", fill="white")
        self.writer.append_data(np.asarray(self.canvas))
        self.frames += 1
        # Match first-success metrics; later budget steps still run unchanged.
        self.finished = successful


@hydra.main(version_base=None, config_path=str(CONFIG_ROOT / "eval"), config_name="ant_l2")
def run(cfg: DictConfig) -> None:
    cfg = storage_config(cfg)
    if "clips" in cfg:
        raise ValueError("eval_once records simulated execution; use eval.py for DROID offline clips")
    if not cfg.get("load_eval_trajs_path") or cfg.get("dump_eval_trajs_path"):
        raise ValueError("eval_once requires load_eval_trajs_path pointing to saved evaluation tasks")
    task_index = int(cfg.get("task_index", 0))
    fps = int(cfg.get("video_fps", 10))
    if task_index < 0 or fps <= 0:
        raise ValueError("task_index must be nonnegative and video_fps must be positive")
    # Enforce the single-task contract for library callers as well as the CLI.
    cfg = cast(DictConfig, OmegaConf.merge(cfg, {
        "task_index": task_index, "video_fps": fps,
        "world": {"num_envs": 1}, "eval": {"num_eval": 1, "chunk_size": 1},
    }))
    output_dir = resolve_results_dir(cfg)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"task_{task_index:03d}.mp4"
    pending_path = video_dir / f".task_{task_index:03d}.pending.mp4"

    import imageio.v2 as imageio

    try:
        with cast(_VideoWriter, imageio.get_writer(pending_path, fps=fps, codec="libx264")) as writer:
            video = _ExecutionVideo(writer, task_index, int(cfg.eval.eval_budget))
            metrics = run_planning_eval(cfg, task_index=task_index, on_step=video)
        pending_path.replace(video_path)
    finally:
        pending_path.unlink(missing_ok=True)
    metrics.update({
        "task_index": task_index,
        "video": str(video_path.relative_to(output_dir)),
        "video_fps": fps,
        "video_frames": video.frames,
    })
    OmegaConf.save(OmegaConf.create(metrics), output_dir / "metrics.yaml")
    print(f"Task {task_index}: success={bool(metrics['episode_successes'][0])}; video={video_path}")
