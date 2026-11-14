# uuid
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf
from stable_pretraining.callbacks.hf_models import HuggingFaceCheckpointCallback
from torch.utils.data import DataLoader

from h_jepa.data import (
    Compose,
    build_hdf5_dataset,
    build_normalizer_artifact,
    get_column_normalizer,
    save_normalizer_artifact,
)
from h_jepa.final_probing_decoding_eval import FinalProbingDecodingEvalCallback
from h_jepa.hjepa_utils import create_world_model, hjepa_forward
from h_jepa.runtime import storage_config
from h_jepa.utils import (
    DebugArtifactCleanupCallback,
    ModelObjectCallBack,
    PlanningEvalCallback,
    ResumeCheckpoint,
)
from stable_worldmodel.paths import CONFIG_ROOT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GradClipModule(spt.Module):
    _train_cfg: DictConfig
    def clip_gradients(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        if gradient_clip_val is None or not self._train_cfg.get("grad_clip_per_level", False):
            return super().clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)
        # grad_clip_per_level: each param group (level{N}) gets its own norm budget.
        for group in optimizer.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if params:
                torch.nn.utils.clip_grad_norm_(params, float(gradient_clip_val))


class ResumableManager(spt.Manager):
    """spt.Manager that resumes from the run dir's `ResumeCheckpoint` file.

    With `resume_file` set, any start (new job id or SLURM requeue) loads it with full state
    when it exists and otherwise starts fresh (from `ckpt_path` if given). spt's own requeue
    index is never used then: it is keyed by SLURM job id, so a second run inside the same job
    would restore the first run's state. With `resume_file=None`, spt decides, except that a
    requeue preempted before spt's first epoch-end checkpoint starts fresh instead of raising.
    """

    def __init__(self, *args, resume_file: Path | None, model_dir: Path, **kwargs):
        self.model_dir = Path(model_dir)
        super().__init__(*args, **kwargs)
        self.resume_file = None if resume_file is None else Path(resume_file)

    def _configure_cache_dir_checkpointing(self):
        super()._configure_cache_dir_checkpointing()
        # SPT keeps registry/logging metadata in outputs; model state is separate.
        checkpoint_dir = self.model_dir / "spt" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for callback in self._trainer.callbacks:
            if isinstance(callback, pl.pytorch.callbacks.ModelCheckpoint):
                callback.dirpath = str(checkpoint_dir)
            elif isinstance(callback, HuggingFaceCheckpointCallback):
                callback.save_dir = self.model_dir / "spt" / "hf_exports"

    def _resolve_load_path(self, run_dir):
        if self.resume_file is not None:
            if self.resume_file.is_file():
                logging.info(f"Resuming full training state from {self.resume_file}")
                return str(self.resume_file), False
            logging.info(f"No resume file at {self.resume_file}; starting from ckpt_path={self.ckpt_path}")
            return (str(self.ckpt_path), self.weights_only) if self.ckpt_path else (None, None)
        try:
            return super()._resolve_load_path(self.model_dir / "spt")
        except RuntimeError as e:
            if "no last.ckpt" not in str(e):
                raise
            logging.warning(f"Requeue with nothing to resume from; starting fresh ({e})")
            return None, None


def _build_hjepa_optimizer_factory(model, cfg):
    optimizer_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.optimizer, resolve=True))
    optimizer_cfg.pop("warmup_ratio", None)

    def optimizer_factory(params):
        param_groups = []
        for level in range(1, int(cfg.num_levels) + 1):
            level_params = [param for param in model.get_level(level).parameters() if param.requires_grad]
            param_groups.append(
                {
                    "params": level_params,
                    "lr": cfg[f"level{level}"].lr,
                    "name": f"level{level}",
                }
            )

        rows = [(group["name"], group["lr"], len(group["params"])) for group in param_groups]
        logging.info(f"HJEPA optimizer parameter groups: {rows}")
        return spt.optim.create_optimizer(param_groups, optimizer_cfg)

    return optimizer_factory


def _configure_runtime_performance():
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _normalize_uint8_images_on_device(batch):
    for key, value in list(batch.items()):
        if not key.startswith("pixels"):
            continue
        if not torch.is_tensor(value) or value.dtype != torch.uint8:
            continue
        if value.ndim < 4 or value.size(-3) != 3:
            raise ValueError(
                f"Expected uint8 pixel tensor '{key}' to have channel-first RGB "
                f"shape (..., 3, H, W), got {tuple(value.shape)}"
            )

        view_shape = [1] * value.ndim
        view_shape[-3] = 3
        mean = value.new_tensor(IMAGENET_MEAN, dtype=torch.float32).view(*view_shape)
        std = value.new_tensor(IMAGENET_STD, dtype=torch.float32).view(*view_shape)
        batch[key] = value.float().div(255.0).sub(mean).div(std)


def _hjepa_training_forward(self, batch, stage, cfg):
    return hjepa_forward(
        self,
        batch,
        stage,
        cfg,
        normalize_batch=_normalize_uint8_images_on_device,
    )


@hydra.main(version_base=None, config_path=str(CONFIG_ROOT / "train"), config_name=None)
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    _configure_runtime_performance()
    cfg = storage_config(cfg)
    cache_dir = cfg.datasets

    dataset_cfg = {k: v for k, v in cfg.data.dataset.items() if k != "val_name"}
    val_total_transitions = dataset_cfg.pop("val_total_transitions", None)
    val_name = cfg.data.dataset.get("val_name", None)
    train_dataset = build_hdf5_dataset(dataset_cfg, cache_dir=cache_dir)

    val_dataset_cfg = {k: v for k, v in dataset_cfg.items() if k not in ("total_transitions", "data_path")}
    val_dataset_cfg["name"] = val_name
    if val_total_transitions is not None:
        val_dataset_cfg["total_transitions"] = int(val_total_transitions)
    val_dataset = build_hdf5_dataset(val_dataset_cfg, cache_dir=cache_dir)

    # Stored images already match img_size; uint8 pixels are normalized on device.
    extra_transforms = [
        get_column_normalizer(train_dataset, col, col)
        for col in cfg.data.dataset.keys_to_load
        if not col.startswith("pixels")
    ]
    transform = Compose(*extra_transforms)
    train_dataset.transform = transform
    val_dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train = DataLoader(
        train_dataset,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.loader, resolve=True))
    val_cfg["shuffle"] = True
    val_cfg["drop_last"] = False
    val = DataLoader(val_dataset, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    normalizer_artifact = build_normalizer_artifact(cfg, train_dataset)
    world_model, losses = create_world_model(cfg)
    world_model.normalizer_artifact = normalizer_artifact
    models = {
        "model": world_model,
    }

    optimizers = {}
    hjepa_optimizer = _build_hjepa_optimizer_factory(world_model, cfg)
    scheduler = {"type": "LinearWarmupCosineAnnealingLR"}
    warmup_ratio = cfg.optimizer.get("warmup_ratio", None)
    if warmup_ratio is not None:

        def scheduler(optimizer, module):
            total_steps = int(module.trainer.estimated_stepping_batches)
            return spt.optim.create_scheduler(
                optimizer,
                {
                    "type": "LinearWarmupCosineAnnealingLR",
                    "warmup_steps": max(1, round(float(warmup_ratio) * total_steps)),
                    "max_steps": total_steps,
                },
                module,
            )

    for model_name in models:
        optimizers[f"{model_name}_opt"] = {
            "modules": str(model_name),
            "optimizer": hjepa_optimizer,
            "scheduler": scheduler,
            "interval": "epoch",
        }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = GradClipModule(
        **models,
        **losses,
        forward=partial(_hjepa_training_forward, cfg=cfg),
        optim=optimizers,
    )
    world_model._train_cfg = cfg
    world_model._output_model_name = cfg.output_model_name

    ##########################
    ##       training       ##
    ##########################

    # run_id = cfg.get("subdir") or ""

    # use rand_str to ensure unique run_dir for each run if subdir is not specified in cfg.
    rand_str = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    run_id = cfg.get("subdir") or rand_str

    run_dir = Path(cfg.models) / run_id
    output_dir = Path(cfg.outputs) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    spt.set(cache_dir=str(output_dir / "spt"))
    logging.info(f"🫆🫆🫆 Run ID: {run_id} 🫆🫆🫆")

    logger = None
    if cfg.wandb.enabled and not cfg.get("quick_debug", False):
        from lightning.pytorch.loggers import WandbLogger

        wandb_config = cast(dict[str, Any], OmegaConf.to_container(cfg.wandb.config, resolve=True))
        wandb_config.setdefault("save_dir", str(output_dir))
        logger = WandbLogger(**wandb_config)
        logger.log_hyperparams(cast(dict[str, Any], OmegaConf.to_container(cfg)))
    else:
        # Without an explicit logger, Lightning defaults the CSVLogger to the cwd,
        # so concurrent jobs sharing one snapshot dir all write the same
        # lightning_logs/version_0/metrics.csv and corrupt each other's header
        # rewrites. Root it at the per-run run_dir instead.
        from lightning.pytorch.loggers import CSVLogger

        logger = CSVLogger(save_dir=str(output_dir), name="lightning_logs")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    normalizer_path = save_normalizer_artifact(normalizer_artifact, run_dir)
    logging.info(f"Saved training normalizer artifact to {normalizer_path}")

    lr_callback = pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=int(cfg.save_every_n_epochs),
    )
    debug_cleanup_callback = DebugArtifactCleanupCallback(
        enabled=bool(cfg.get("quick_debug", False)),
        run_dir=run_dir,
    )
    _pe = cfg.planning_eval
    planning_eval_callback = PlanningEvalCallback(
        enabled=bool(_pe.enabled),
        eval_cfg=_pe,
        run_dir=run_dir,
        storage={name: cfg[name] for name in ("datasets", "models", "outputs")},
        results_dir=output_dir,
        seed=cfg.seed,
        output_subdir=_pe.output_subdir,
        run_on_train_end=bool(_pe.run_on_train_end),
        every_n_epochs=int(_pe.every_n_epochs),
    )
    final_probing_decoding_eval_cfg = cfg.final_probing_decoding_eval
    final_probing_decoding_eval_callback = FinalProbingDecodingEvalCallback(
        eval_cfg=final_probing_decoding_eval_cfg,
        run_dir=run_dir,
        storage={name: cfg[name] for name in ("datasets", "models", "outputs")},
        results_dir=output_dir,
        output_subdir=final_probing_decoding_eval_cfg.get(
            "output_subdir",
            "final_probing_decoding_eval",
        ),
    )

    resume_file = run_dir / "lightning_resume" / "last.ckpt"
    resume_every = cfg.get("resume_every_n_steps", 2000)
    resume_callbacks = [ResumeCheckpoint(resume_file, resume_every)] if resume_every else []
    spt.set(requeue_checkpoint=not resume_every)  # lightning_resume/ already holds the full state

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            *resume_callbacks,
            debug_cleanup_callback,
            object_dump_callback,
            planning_eval_callback,
            final_probing_decoding_eval_callback,
            lr_callback,
        ],
        logger=logger,
        enable_checkpointing=False,
    )

    manager_ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    if cfg.get("quick_debug", False) or not manager_ckpt_path.exists():
        manager_ckpt_path = None

    manager = ResumableManager(
        model_dir=run_dir,
        resume_file=resume_file if resume_every else None,
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=cfg.seed,
        ckpt_path=manager_ckpt_path,
    )

    manager()

