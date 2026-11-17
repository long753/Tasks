#!/usr/bin/env bash
# Run SG-L3, E2E-L3 and E2E-L4 sequentially, one complete epoch each.
# Activate hjepa inside chenlong, then explicitly select two allocated GPUs:
# CUDA_VISIBLE_DEVICES=6,7 bash scripts/train/train_fourroom_ep1.sh --run-name fourroom_ep1_compare
# Append --cfg job --resolve to inspect all three configurations without training.
# --run-name is a prefix; storage arguments are forwarded to train.py.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
RUN_NAME_PREFIX="fourroom_ep1_compare"
ARGS=()
while (($#)); do
  case $1 in
    --run-name)
      (($# > 1)) || { echo "missing value for --run-name" >&2; exit 2; }
      RUN_NAME_PREFIX=$2
      shift 2
      ;;
    --run-name=*)
      RUN_NAME_PREFIX=${1#*=}
      shift
      ;;
    -h|--help)
      cat <<'HELP'
Usage: CUDA_VISIBLE_DEVICES=6,7 bash scripts/train/train_fourroom_ep1.sh [arguments]
Run inside chenlong after activating conda environment hjepa.

  --run-name PREFIX     Experiment prefix (default: fourroom_ep1_compare)
  --datasets PATH      Dataset root (default: project's datasets/)
  --models PATH        Model root (default: project's models/)
  --outputs PATH       Result root (default: project's outputs/)
  --cfg job --resolve  Print all three configurations without training

Requires exactly two explicitly selected, allocated GPUs. Jobs run sequentially:
PREFIX/sg_l3_seed42, PREFIX/e2e_l3_seed42, PREFIX/e2e_l4_seed42.
Each uses one epoch, batch 32 per GPU, bf16, zero loader workers, and no post-training eval.
Uses a fresh one-epoch LR schedule, not the first epoch of the old seven-epoch SG-L4 schedule.
Existing run names use train.py's full-state resume behavior; use a new prefix for a fresh experiment.
HELP
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to exactly two allocated GPUs, e.g. CUDA_VISIBLE_DEVICES=6,7." >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if ((${#GPU_IDS[@]} != 2)) || [[ -z ${GPU_IDS[0]} || -z ${GPU_IDS[1]} || ${GPU_IDS[0]} == "-1" || ${GPU_IDS[1]} == "-1" || ${GPU_IDS[0]} == "${GPU_IDS[1]}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must select two distinct GPUs (indices or GPU UUIDs)." >&2
  exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
cd "$ROOT"

COMMON=(
  seed=42
  trainer.devices=2
  trainer.max_epochs=1
  trainer.precision=bf16-mixed
  loader.batch_size=32
  num_workers=0
  loader.persistent_workers=false
  loader.prefetch_factor=null
  planning_eval.enabled=false
  final_probing_decoding_eval.enabled=false
)

for experiment in sg:3 e2e:3 e2e:4; do
  mode=${experiment%%:*}
  levels=${experiment#*:}
  run_name="${RUN_NAME_PREFIX%/}/${mode}_l${levels}_seed42"
  printf '\n=== %s | GPUs=%s | one epoch ===\n' "$run_name" "$CUDA_VISIBLE_DEVICES"
  "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    "$ROOT/scripts/train/train.py" "${ARGS[@]}" \
    --config "fourroom_hjepa_l${levels}" \
    --training-mode "$mode" \
    --run-name "$run_name" \
    "${COMMON[@]}"
done
