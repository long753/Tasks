"""FourRoom evaluation protocol and fixed-task restoration."""

from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from stable_worldmodel.paths import CONFIG_ROOT


def load_fourroom_protocol(datasets: Path) -> DictConfig:
    protocol = cast(DictConfig, OmegaConf.load(CONFIG_ROOT / 'fourroom.yaml'))
    protocol.datasets = str(datasets)
    protocol.world.max_episode_steps = 2 * int(protocol.eval.eval_budget)
    return protocol


def fixed_task_payload(tasks: dict[str, Any], task_index: int) -> dict[str, Any]:
    episode = dict(tasks['data'][task_index])
    env_seed = int(tasks['task_info']['tasks'][task_index]['seed'])
    # World reads seed from episode columns. A scalar tensor row survives its column conversion.
    episode['seed'] = torch.full((len(episode['pixels']),), env_seed, dtype=torch.int64)
    return {
        'data': [episode], 'episodes_idx': [int(tasks['episodes_idx'][task_index])],
        'columns': [*tasks['columns'], 'seed'],
    }
