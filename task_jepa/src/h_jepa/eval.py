import hydra
from omegaconf import DictConfig

from h_jepa.droid_eval import run_clip_eval
from h_jepa.planning_eval import evaluate_prefixes, run_planning_eval
from h_jepa.runtime import storage_config
from stable_worldmodel.paths import CONFIG_ROOT


@hydra.main(version_base=None, config_path=str(CONFIG_ROOT / 'eval'), config_name='pusht_flat')
def run(cfg: DictConfig):
    cfg = storage_config(cfg)
    if (levels := cfg.get('prefix_levels')) is not None:
        evaluate_prefixes(cfg, levels)
    elif 'clips' in cfg:  # DROID offline clip eval
        run_clip_eval(cfg)
    else:
        run_planning_eval(cfg)

