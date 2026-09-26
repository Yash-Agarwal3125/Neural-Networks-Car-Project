#!/bin/bash
# Matched-budget fix: retrain the ordinal-encoding nominal arm at 300
# episodes (matching the angle-tagged arm) so the ordinal-vs-angle-tagged
# comparison is no longer confounded by training-episode budget. Writes to
# a separate out_dir so the paper's already-published 120-episode ordinal
# baseline numbers (Table II, Table III, Fig. 2-5) are untouched.
set -e
cd "$(dirname "$0")/.."
OUT=experiments/full_runs_ordinal_300ep
PY="/c/Users/Vidhi Jain/AppData/Local/Programs/Python/Python310/python.exe"

for seed in 0 1 2; do
  run_dir="$OUT/sensing_train/nominal_seed${seed}"
  if [ -f "$run_dir/final.weights.h5" ]; then
    echo "SKIP (exists) $run_dir"
  else
    echo "=== TRAIN ordinal nominal seed=$seed (300 episodes) ==="
    "$PY" experiments/full_study.py --phase sensing_train --regime nominal --seed "$seed" \
      --episodes 300 --max_steps 350 --encoding ordinal --out_dir "$OUT"
  fi
done

for seed in 0 1 2; do
  eval_csv="$OUT/sensing_eval/nominal_seed${seed}.csv"
  if [ -f "$eval_csv" ]; then
    echo "SKIP (exists) $eval_csv"
  else
    echo "=== EVAL ordinal nominal seed=$seed ==="
    "$PY" experiments/full_study.py --phase sensing_eval --regime nominal --seed "$seed" \
      --eval_episodes 15 --max_steps 350 --encoding ordinal --out_dir "$OUT"
  fi
done

echo "=== ORDINAL MATCHED-BUDGET (300ep) RETRAIN DONE ==="
