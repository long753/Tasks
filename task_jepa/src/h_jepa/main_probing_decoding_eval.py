import hydra

from h_jepa.final_probing_decoding_eval import run_final_probing_decoding_eval
from h_jepa.runtime import storage_config
from stable_worldmodel.paths import CONFIG_ROOT


@hydra.main(version_base=None, config_path=str(CONFIG_ROOT / 'probing'), config_name=None)
def run(cfg):
    run_final_probing_decoding_eval(storage_config(cfg))

