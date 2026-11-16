"""Resolve command storage parameters without environment-variable fallbacks."""

from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from stable_worldmodel.paths import StoragePaths, resolve_path, resolve_run_name


def storage_config(cfg: Any) -> DictConfig:
    paths = StoragePaths.resolve(*(cfg.get(name, name) for name in ("datasets", "models", "outputs")))
    values = {
        **paths.as_dict(),
        "run_name": resolve_run_name(cfg.get("run_name"), prefix=str(cfg.get("model_name", "run"))),
    }
    checkpoint = cfg.get("checkpoint")
    if checkpoint is not None:
        values["checkpoint"] = str(resolve_path(checkpoint))
    return cast(DictConfig, OmegaConf.merge(cfg, values))
