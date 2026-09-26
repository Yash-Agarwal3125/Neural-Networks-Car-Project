#!/bin/bash
# Workstream A: angle-invariant encoding sensing sweep.
# Trains D3QN+PER under the new canonical-angle-grid encoding across the
# same 13-point sensing grid x {nominal, randomized} regime x 3 seeds as the
# paper's already-published baseline sweep, then evaluates. Self-skipping:
# safe to re-run after an interruption.
set -e
cd "$(dirname "$0")/.."
OUT=experiments/full_runs_angle_tagged
PY="/c/Users/Vidhi Jain/AppData/Local/Programs/Python/Python310/python.exe"

for regime in nominal randomized; do
  for seed in 0 1 2; do
    run_dir="$OUT/sensing_train/${regime}_seed${seed}"
    if [ -f "$run_dir/final.weights.h5" ]; then
      echo "SKIP (exists) $run_dir"
    else
      echo "=== TRAIN $regime seed=$seed ==="
      "$PY" experiments/full_study.py --phase sensing_train --regime "$regime" --seed "$seed" \
        --episodes 120 --max_steps 350 --encoding angle_tagged --out_dir "$OUT"
    fi
  done
done

for regime in nominal randomized; do
  for seed in 0 1 2; do
    eval_csv="$OUT/sensing_eval/${regime}_seed${seed}.csv"
    if [ -f "$eval_csv" ]; then
      echo "SKIP (exists) $eval_csv"
    else
      echo "=== EVAL $regime seed=$seed ==="
      "$PY" experiments/full_study.py --phase sensing_eval --regime "$regime" --seed "$seed" \
        --eval_episodes 15 --max_steps 350 --encoding angle_tagged --out_dir "$OUT"
    fi
  done
done

echo "=== ANGLE-TAGGED SWEEP DONE ==="
