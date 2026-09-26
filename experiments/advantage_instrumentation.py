"""
Post-hoc value/advantage-stream instrumentation for the D3QN+PER policies
already trained by `full_study.py --phase sensing_train` (paper.tex Intro
contribution #4 / Section 3 "Placement of this work": "We instrument value
and advantage streams across sensing sweeps to test whether advantage
dispersion contracts prior to performance collapse").

This does NOT retrain anything: it loads the existing final.weights.h5 for
each (regime, seed) sensing_train run into the same dueling architecture,
then builds a diagnostic sibling model that taps the pre-aggregation value
and advantage Dense layers directly (same weights, same forward pass, just
exposing two extra outputs) -- so the instrumentation is exact, not a
re-estimate.

For each of the 13 sensing-grid points, runs a short greedy rollout and
records, per step, the advantage vector's dispersion (std across the 4
actions). Reports mean advantage-dispersion per grid point alongside the
already-known mean-checkpoints performance from sensing_eval, to test
whether dispersion contracts at the grid points where performance collapses
(beam count != 3) versus where it degrades gracefully (FOV/noise/dropout).

Usage:
    python experiments/advantage_instrumentation.py
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import pandas as pd
from keras import Model

import full_study as fs

RUNS_DIR = "experiments/full_runs"
OUT_DIR = os.path.join(RUNS_DIR, "summary")
SEEDS = (0, 1, 2)
EPISODES_PER_POINT = 8
MAX_STEPS = 350


def diagnostic_model(agent_model):
    """Build a sibling model exposing the pre-aggregation value (units=1)
    and advantage (units=action_size, Dense not Lambda) layers, sharing the
    exact same weights/graph as agent_model."""
    value_layer, advantage_layer = None, None
    for layer in agent_model.layers:
        if "dense" not in layer.name:
            continue
        units = int(layer.output.shape[-1])
        if units == 1:
            value_layer = layer
        elif units == 4:
            advantage_layer = layer
    assert value_layer is not None and advantage_layer is not None, \
        "could not locate value/advantage Dense layers in the saved model"
    return Model(inputs=agent_model.input,
                 outputs=[value_layer.output, advantage_layer.output])


def rollout_with_advantage(diag_model, sensing_point, seed, episodes, max_steps):
    fs.set_global_seed(seed + 9000)
    env = fs.SensingGameEnv(max_steps=max_steps, sensing=sensing_point)
    dispersions, checkpoints_list = [], []
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_disp = []
        while not done:
            v, a = diag_model(np.expand_dims(state, 0), training=False)
            a = a.numpy()[0]
            ep_disp.append(float(np.std(a)))
            action = int(np.argmax(v.numpy()[0][0] + (a - a.mean())))
            state, reward, done, info = env.step(action)
        dispersions.append(np.mean(ep_disp) if ep_disp else float("nan"))
        checkpoints_list.append(env.checkpoints_cleared)
    return float(np.mean(dispersions)), float(np.mean(checkpoints_list))


def main():
    rows = []
    for regime in ("nominal", "randomized"):
        for seed in SEEDS:
            weights_path = os.path.join(RUNS_DIR, "sensing_train", f"{regime}_seed{seed}",
                                         "final.weights.h5")
            if not os.path.exists(weights_path):
                print(f"WARNING: missing {weights_path}, skipping")
                continue
            cfg = fs.ABLATION_CONFIGS["d3qn_per"]
            agent = fs.Agent(fs.MAX_BEAMS + 2, 4, **cfg)
            agent.model.load_weights(weights_path)
            diag = diagnostic_model(agent.model)
            for point in fs.SENSING_GRID:
                disp, mean_cp = rollout_with_advantage(
                    diag, point, seed, EPISODES_PER_POINT, MAX_STEPS)
                rows.append(dict(regime=regime, seed=seed, mean_advantage_std=disp,
                                  mean_checkpoints=mean_cp, **point))
            print(f"done {regime} seed {seed}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "advantage_dispersion_raw.csv"), index=False)

    factors = {
        "beam_count": (1, 3, 5, 7), "fov_deg": (60, 90, 120, 150),
        "noise_sigma": (0.0, 0.02, 0.05, 0.1), "dropout_p": (0.0, 0.1, 0.2, 0.4),
    }
    nominal_point = dict(beam_count=3, fov_deg=90.0, noise_sigma=0.0, dropout_p=0.0)
    summary_rows = []
    for factor, levels in factors.items():
        others = {k: v for k, v in nominal_point.items() if k != factor}
        for level in levels:
            mask = (df[factor] == level)
            for k, v in others.items():
                mask &= np.isclose(df[k].astype(float), float(v))
            for regime in ("nominal", "randomized"):
                sub = df[mask & (df["regime"] == regime)]
                if len(sub) == 0:
                    continue
                summary_rows.append(dict(
                    factor=factor, level=level, regime=regime,
                    mean_advantage_std=sub["mean_advantage_std"].mean(),
                    mean_checkpoints=sub["mean_checkpoints"].mean(),
                ))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "advantage_dispersion_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
