#!/bin/bash
set -e
PY="/c/Users/Vidhi Jain/AppData/Local/Programs/Python/Python310/python.exe"
cd "$(dirname "$0")/.."
echo "=== $(date) resume: sensing_train remaining runs ==="
for pair in "nominal 1" "nominal 2" "randomized 0" "randomized 1" "randomized 2"; do
  set -- $pair
  regime=$1; seed=$2
  # skip if already fully done (final.weights.h5 exists)
  if [ -f "experiments/full_runs/sensing_train/${regime}_seed${seed}/final.weights.h5" ]; then
    echo "=== $(date) SKIP $regime seed $seed (already done) ==="
    continue
  fi
  echo "=== $(date) starting sensing_train $regime seed $seed ==="
  "$PY" experiments/full_study.py --phase sensing_train --regime "$regime" --seed "$seed" --out_dir experiments/full_runs
done
echo "=== $(date) sensing_train remaining DONE, starting sensing_eval ==="
for regime in nominal randomized; do
  for seed in 0 1 2; do
    if [ -f "experiments/full_runs/sensing_eval/${regime}_seed${seed}.csv" ]; then
      echo "=== $(date) SKIP eval $regime seed $seed (already done) ==="
      continue
    fi
    echo "=== $(date) starting sensing_eval $regime seed $seed ==="
    "$PY" experiments/full_study.py --phase sensing_eval --regime "$regime" --seed "$seed" --out_dir experiments/full_runs
  done
done
echo "=== $(date) ALL REMAINING WORK DONE ==="
