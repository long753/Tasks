"""Greedy stage orchestration and durable per-level training checkpoints."""

import gc
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from h_jepa.data import load_normalizer_artifact
from h_jepa.final_probing_decoding_eval import FinalProbingDecodingEvalCallback
from h_jepa.hjepa_utils import create_world_model
from h_jepa.runtime import storage_config
from h_jepa.utils import PlanningEvalCallback, ResumeCheckpoint


def _atomic_save(path: Path, value: Any, *, json_format: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            if json_format:
                stream.write(json.dumps(value).encode("utf-8"))
            else:
                torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _stage_contract(cfg: DictConfig, level: int) -> dict[str, int]:
    return {
        "level": level,
        "num_levels": int(cfg.num_levels),
        "max_epochs": int(cfg.trainer.max_epochs),
        "max_steps": int(cfg.trainer.get("max_steps", -1)),
    }


class LayerwiseStage(ResumeCheckpoint):
    """Commit completion only after the full stage state and final exports succeed."""

    def __init__(self, cfg: DictConfig, level: int):
        self.cfg = cfg
        self.level = level
        self.root = Path(cfg.models) / cfg.run_name
        self.directory = self.root / "stages" / f"level{level}"
        super().__init__(self.directory / "lightning_resume" / "last.ckpt", cfg.get("resume_every_n_steps", 2000))
        self.contract = _stage_contract(cfg, level)
        self.complete = False

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero and not (self.root / "normalizer.pt").exists():
            _atomic_save(self.root / "normalizer.pt", pl_module.model.normalizer_artifact)
        trainer.strategy.barrier("layerwise_normalizer")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["layerwise_stage"] = {**self.contract, "complete": self.complete}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        metadata = checkpoint.get("layerwise_stage", {})
        if any(metadata.get(key) != value for key, value in self.contract.items()):
            raise ValueError("Layerwise checkpoint stage or training budget does not match this run")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n > 0:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def _export_final(self, trainer, pl_module):
        if trainer.is_global_zero:
            _atomic_save(self.root / f"{self.cfg.model_name}_object.ckpt", pl_module.model)
        trainer.strategy.barrier("layerwise_final_model")
        storage = {name: self.cfg[name] for name in ("datasets", "models", "outputs")}
        results = Path(self.cfg.outputs) / self.cfg.run_name
        planning = self.cfg.planning_eval
        callbacks = [
            PlanningEvalCallback(
                enabled=bool(planning.enabled), eval_cfg=planning,
                run_dir=self.root, storage=storage, results_dir=results,
                seed=self.cfg.seed, output_subdir=planning.output_subdir,
                run_on_train_end=bool(planning.run_on_train_end), every_n_epochs=0,
            ),
            FinalProbingDecodingEvalCallback(
                eval_cfg=self.cfg.final_probing_decoding_eval, run_dir=self.root,
                storage=storage, results_dir=results,
                output_subdir=self.cfg.final_probing_decoding_eval.get("output_subdir", "final_probing_decoding_eval"),
            ),
        ]
        stage_cfg = pl_module._train_cfg
        try:
            pl_module._train_cfg = self.cfg
            for callback in callbacks:
                callback.on_train_end(trainer, pl_module)
        finally:
            pl_module._train_cfg = stage_cfg

    def on_fit_end(self, trainer, pl_module):
        steps_done = trainer.max_steps > 0 and trainer.global_step >= trainer.max_steps
        epochs_done = (
            trainer.max_epochs is not None and trainer.max_epochs > 0
            and trainer.current_epoch >= trainer.max_epochs and not trainer.should_stop
        )
        if trainer.interrupted or not (steps_done or epochs_done):
            raise RuntimeError(f"Layerwise level {self.level} stopped before its training budget was completed")
        if self.level == int(self.cfg.num_levels):
            self._export_final(trainer, pl_module)
        self.complete = True
        self._save(trainer)
        trainer.strategy.barrier("layerwise_stage_checkpoint")
        if trainer.is_global_zero:
            _atomic_save(
                self.directory / "progress.json",
                {**self.contract, "complete": True, "global_step": int(trainer.global_step)},
                json_format=True,
            )
        trainer.strategy.barrier("layerwise_stage_committed")


def run_layerwise_training(cfg: DictConfig, train_stage: Callable[..., Any]) -> None:
    cfg = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(storage_config(cfg), resolve=True)))
    if int(cfg.trainer.max_epochs) <= 0:
        raise ValueError("Layerwise training requires a positive trainer.max_epochs budget per level")
    root = Path(cfg.models) / cfg.run_name
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.yaml"
    if config_path.exists() and OmegaConf.load(config_path) != cfg:
        raise ValueError("Layerwise run configuration changed; use a new --run-name for different training settings")
    if not config_path.exists():
        # Resolve first: pruning future dataset levels must not invalidate model interpolations.
        with tempfile.NamedTemporaryFile(dir=root, prefix=".config.", delete=False, mode="w") as stream:
            temporary = Path(stream.name)
            try:
                OmegaConf.save(cfg, stream)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(config_path)
        finally:
            temporary.unlink(missing_ok=True)

    completed = 0
    missing = False
    for level in range(1, int(cfg.num_levels) + 1):
        progress = root / "stages" / f"level{level}" / "progress.json"
        if not progress.exists():
            missing = True
            continue
        metadata = json.loads(progress.read_text())
        if any(metadata.get(key) != value for key, value in _stage_contract(cfg, level).items()):
            raise ValueError(f"Invalid progress metadata for layerwise level {level}")
        if not metadata.get("complete"):
            missing = True
            continue
        if missing:
            raise ValueError("Layerwise completed stages are not contiguous")
        checkpoint = progress.parent / "lightning_resume" / "last.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Completed layerwise stage is missing its checkpoint: {checkpoint}")
        completed = level

    if completed == int(cfg.num_levels):
        if not (root / f"{cfg.model_name}_object.ckpt").is_file():
            raise FileNotFoundError("Completed layerwise run is missing its final model")
        return

    pl.seed_everything(int(cfg.seed), workers=True)
    model, losses = create_world_model(cfg)
    if completed:
        checkpoint = root / "stages" / f"level{completed}" / "lightning_resume" / "last.ckpt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected = _stage_contract(cfg, completed)
        metadata = state.get("layerwise_stage", {})
        if not metadata.get("complete") or any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Completed layerwise checkpoint does not match its progress metadata")
        weights = state["state_dict"]
        model.load_state_dict({key.removeprefix("model."): value for key, value in weights.items() if key.startswith("model.")})
        for name, regularizer in losses.items():
            prefix = f"{name}."
            regularizer.load_state_dict({key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)})
        del state, weights
    if (root / "normalizer.pt").exists():
        model.normalizer_artifact = load_normalizer_artifact(root / "normalizer.pt")

    for level in range(completed + 1, int(cfg.num_levels) + 1):
        stage_cfg = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
        with open_dict(stage_cfg):
            stage_cfg.run_name = f"{cfg.run_name}/stages/level{level}"
            for future in range(level + 1, int(cfg.num_levels) + 1):
                stage_cfg.data.dataset.pop(f"level{future}", None)
        callback = LayerwiseStage(cfg, level)
        model = train_stage(
            stage_cfg, detach_lower_state=True, active_level=level,
            world_model=model, losses=losses, stage_callback=callback,
        )
        if not callback.complete:
            raise RuntimeError(f"Layerwise level {level} did not commit a completed checkpoint")
        gc.collect()
