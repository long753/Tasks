#!/usr/bin/env bash
# Collect the eight Visual AntMaze and FourRoom Distractors datasets into the
# selected --datasets directory. AntMaze expects its expert checkpoint at
# <datasets>/ogbench_experts/ant/. Re-running resumes interrupted collection.
set -e
cd "$(dirname "$0")"

ANT=${ANT:-"explore_stitch_train stitch_val probing_train probing_eval"}
FOURROOM=${FOURROOM:-"fourroom_tp35_d1 fourroom_tp35_d1_val fourroom_tp35_d1_probing fourroom_tp35_d1_probing_val"}

for name in $ANT; do
  python collect_antmaze.py --config-name "visual_antmaze_medium_$name" "$@"
done
for name in $FOURROOM; do
  python collect_fourroom_distractors.py --config-name "$name" "$@"
done
