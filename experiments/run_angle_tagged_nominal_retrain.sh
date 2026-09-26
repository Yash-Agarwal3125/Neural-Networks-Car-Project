#!/bin/bash
# Retrain only the angle-tagged nominal-trained arm with a longer episode
# budget (300 vs the original 120), since the larger 33-dim canonical-angle
# state vector failed to converge at all under the original matched-budget
# setting (0 checkpoints across all 3 seeds). Randomized-trained arm is left
# untouched (it converged fine at 120 episodes).
set -e
cd "$(dirname "$0")/.."
OUT=experiments/full_runs_angle_tagged
PY="/c/Users/Vidhi Jain/AppData/Local/Programs/Python/Python310/python.exe"

for seed in 0 1 2; do
  run_dir="$OUT/sensing_train/nominal_seed${seed}"
  if [ -f "$run_dir/final.weights.h5" ]; then
    echo "SKIP (exists) $run_dir"
  else
    echo "=== TRAIN nominal seed=$seed (300 episodes) ==="
    "$PY" experiments/full_study.py --phase sensing_train --regime nominal --seed "$seed" \
      --episodes 300 --max_steps 350 --encoding angle_tagged --out_dir "$OUT"
  fi
done

for seed in 0 1 2; do
  eval_csv="$OUT/sensing_eval/nominal_seed${seed}.csv"
  if [ -f "$eval_csv" ]; then
    echo "SKIP (exists) $eval_csv"
  else
    echo "=== EVAL nominal seed=$seed ==="
    "$PY" experiments/full_study.py --phase sensing_eval --regime nominal --seed "$seed" \
      --eval_episodes 15 --max_steps 350 --encoding angle_tagged --out_dir "$OUT"
  fi
done

echo "=== NOMINAL RETRAIN DONE ==="
