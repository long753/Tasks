"""Project-relative storage locations shared by commands and library callers."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class StoragePaths:
    datasets: Path
    models: Path
    outputs: Path

    @classmethod
    def resolve(
        cls,
        datasets: str | Path = "datasets",
        models: str | Path = "models",
        outputs: str | Path = "outputs",
    ) -> "StoragePaths":
        return cls(*(resolve_path(value) for value in (datasets, models, outputs)))

    def as_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in ("datasets", "models", "outputs")}


def add_storage_arguments(parser: argparse.ArgumentParser, *, include_models: bool = True) -> None:
    names = ("datasets", "models", "outputs") if include_models else ("datasets", "outputs")
    for name in names:
        parser.add_argument(f"--{name}", default=name, help=f"{name} directory (relative to the project root)")


def resolve_run_name(value: str | None = None, *, prefix: str = "run") -> str:
    """Create a unique default name, or validate a relative experiment namespace."""
    name = value if value is not None else f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    path = Path(name)
    if not name.strip() or path.is_absolute() or not path.parts or ".." in path.parts or path == Path("."):
        raise ValueError("--run-name must be a nonempty relative name without '..'")
    return str(path)


def run_hydra(
    entrypoint: Callable[..., Any],
    argv: Sequence[str] | None = None,
    *,
    command: Literal["train", "eval", "eval_once", "probing", "data"],
) -> None:
    """Expose ordinary command flags; Hydra handles only experiment overrides."""
    parser = argparse.ArgumentParser(
        description=f"{command}: select an experiment with --config; extra key=value arguments tune algorithm settings.",
        allow_abbrev=False,
    )
    if command == "train":
        parser.add_argument(
            "--training-mode", choices=("e2e", "sg", "layerwise"), default="e2e",
            help="e2e (default): joint training; sg: state stop-gradient; layerwise: train and freeze one level at a time",
        )
    if command == "eval":
        parser.add_argument("--prefix-levels", metavar="1,2,3", help="Evaluate proper model prefixes of these depths")
    if command == "eval_once":
        parser.add_argument("--task-index", type=int, default=0, help="Zero-based index in the saved evaluation task file")
        parser.add_argument("--video-fps", type=int, default=10, help="Playback frames per second")
    add_storage_arguments(parser, include_models=command not in ("eval", "eval_once"))
    parser.add_argument("--config", required=command != "data", help="Experiment configuration name, without .yaml")
    parser.add_argument("--run-name", help="Experiment name; defaults to the config name plus a unique timestamp")
    if command in ("eval", "eval_once", "probing"):
        parser.add_argument("--checkpoint", required=True, help="Checkpoint file; relative paths start at the project root")
    parser.add_argument("--cfg", choices=("job", "hydra", "all"), help="Print configuration without running")
    parser.add_argument("--resolve", action="store_true", help="Resolve interpolations when printing configuration")
    args, overrides = parser.parse_known_args(argv)
    obsolete = {
        "policy": "--checkpoint",
        "checkpoint": "--checkpoint",
        "output": "--outputs and --run-name",
        "output.dir": "--outputs and --run-name",
        "output_dir": "--outputs and --run-name",
        "output_subdir": "--run-name",
        "output_model_name": "--run-name",
        "subdir": "--run-name",
        "run_name": "--run-name",
        "datasets": "--datasets",
        "models": "--models",
        "outputs": "--outputs",
        "hydra.run.dir": "--outputs and --run-name",
        "hydra.sweep.dir": "--outputs and --run-name",
    }
    if command == "train":
        obsolete.update({
            "training_mode": "--training-mode",
            "training_variant": "--training-mode",
            "detach_lower_state": "--training-mode",
        })
    if command == "eval":
        obsolete["prefix_levels"] = "--prefix-levels"
    if command == "data":
        # Data collection policy config is an expert policy, not a model input.
        obsolete.pop("policy")
    if command == "eval_once":
        if args.task_index < 0 or args.video_fps <= 0:
            parser.error("--task-index must be nonnegative and --video-fps must be positive")
        obsolete.update({"task_index": "--task-index", "video_fps": "--video-fps"})
    for item in overrides:
        if item.startswith("-") or ("=" not in item and not item.startswith("~")):
            parser.error(f"unrecognized argument: {item}")
        key = item.lstrip("+~").partition("=")[0]
        if key in obsolete:
            parser.error(f"{key} is no longer a command override; use {obsolete[key]}")
        if command == "eval_once" and key in ("eval.num_eval", "world.num_envs", "eval.chunk_size"):
            parser.error("eval_once always runs one task; use eval.py for multi-task evaluation")
    paths = StoragePaths.resolve(args.datasets, getattr(args, "models", "models"), args.outputs)
    try:
        name = resolve_run_name(args.run_name, prefix=args.config or command)
    except ValueError as error:
        parser.error(str(error))
    values: dict[str, str | int | list[int]] = {**paths.as_dict(), "run_name": name}
    if command == "eval" and args.prefix_levels is not None:
        try:
            values["prefix_levels"] = [int(depth) for depth in args.prefix_levels.split(",")]
        except ValueError:
            parser.error("--prefix-levels must be a comma-separated list of integers, e.g. 1,2,3")
    if command == "train":
        detach_lower_state = args.training_mode != "e2e"
        values.update({
            "training_variant": {
                "e2e": "end_to_end", "sg": "state_stop_gradient", "layerwise": "layerwise",
            }[args.training_mode],
            "detach_lower_state": detach_lower_state,
        })
    if command in ("eval", "eval_once", "probing"):
        values["checkpoint"] = str(resolve_path(args.checkpoint))
    if command == "eval_once":
        values.update({
            "task_index": args.task_index,
            "video_fps": args.video_fps,
            "eval.num_eval": 1,
            "world.num_envs": 1,
            "eval.chunk_size": 1,
        })
    injected = [f"++{key}={json.dumps(value)}" for key, value in values.items()]
    injected.append(f"hydra.run.dir={json.dumps(str(paths.outputs / name / 'hydra'))}")
    selection = ["--config-name", args.config] if args.config is not None else []
    inspection = ["--cfg", args.cfg] if args.cfg is not None else []
    if args.resolve:
        inspection.append("--resolve")
    original = sys.argv
    sys.argv = [original[0], *selection, *inspection, *overrides, *injected]
    try:
        entrypoint()
    finally:
        sys.argv = original
