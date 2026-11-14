"""Resolve command storage parameters without environment-variable fallbacks."""

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from stable_worldmodel.paths import StoragePaths


def storage_config(cfg: Any) -> DictConfig:
    paths = StoragePaths.resolve(*(cfg.get(name, name) for name in ("datasets", "models", "outputs")))
    return cast(DictConfig, OmegaConf.merge(cfg, paths.as_dict()))


def model_run_id(checkpoint: Path, models: str | Path) -> Path:
    """Keep the model's run hierarchy when writing results outside its directory."""
    try:
        return checkpoint.parent.relative_to(Path(models))
    except ValueError:
        return Path(checkpoint.parent.name)
