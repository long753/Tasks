"""Probing command.

python scripts/eval/probing.py --config ant --checkpoint ./models/ant_l2_exp1/ant_hjepa_l2_object.ckpt --run-name ant_l2_probe
"""

from h_jepa.main_probing_decoding_eval import run
from stable_worldmodel.paths import run_hydra

if __name__ == "__main__":
    run_hydra(run, command="probing")
