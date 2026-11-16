"""Run one saved simulation task and record actual execution beside the goal.

python scripts/eval/eval_once.py --config pusht_l2 \
    --checkpoint ./models/h_jepa/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt \
    --task-index 10
"""

from h_jepa.eval_once import run
from stable_worldmodel.paths import run_hydra

if __name__ == "__main__":
    run_hydra(entrypoint=run, command="eval_once")
