"""Eval command.

python scripts/eval/eval.py \
    --config ant_l2 \
    --checkpoint ./models/h_jepa/ant/ant_hjepa_l2/seed42/ant_hjepa_l2_object.ckpt
"""

from h_jepa.eval import run
from stable_worldmodel.paths import run_hydra

if __name__ == "__main__":
    run_hydra(entrypoint=run, command="eval")
