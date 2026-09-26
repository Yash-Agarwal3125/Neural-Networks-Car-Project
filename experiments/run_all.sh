#!/bin/bash
set -e
PY="/c/Users/Vidhi Jain/AppData/Local/Programs/Python/Python310/python.exe"
cd "$(dirname "$0")/.."
for cfg in dqn double_dqn dueling_dqn d3qn_per; do
  for seed in 0 1 2; do
    echo "=== $(date) starting $cfg seed $seed ==="
    "$PY" experiments/ablation.py --config "$cfg" --seed "$seed" \
      --episodes 120 --max_steps 350 --out_dir experiments/runs
  done
done
echo "=== ALL RUNS DONE $(date) ==="
