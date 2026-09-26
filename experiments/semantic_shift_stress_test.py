"""
Semantic-shift stress tests: strengthens the causal claim that beam-count
brittleness is a REPRESENTATION-mismatch problem, not an information-loss
problem, by testing two shifts that change how information is presented
without necessarily reducing how much information is present.

(1) Beam-order permutation: the same (beam, physical-angle, reading) triples
    are reported in a different array order (e.g. simulating a sensor
    rewiring). For the ordinal encoding this MUST break performance, since
    array position is all that encoding uses. For the angle-tagged encoding
    this predicts a NO-OP by construction: each reading is paired with its
    own true physical angle before assignment, so permuting the arrival
    order of (angle, reading) pairs cannot change which canonical bin each
    reading lands in.

(2) Angular placement shift: the sensor's real field of view is rotated by
    a fixed offset (a genuinely different physical sensor placement, not
    just relabeling). This changes real information (the beams now look in
    different real-world directions), so some degradation is expected for
    BOTH encodings -- reported honestly either way.

Both are eval-only: reuses already-trained weights from
experiments/full_runs/sensing_train (ordinal) and
experiments/full_runs_angle_tagged/sensing_train (angle-tagged), no
retraining.

Usage:
    python experiments/semantic_shift_stress_test.py
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import pandas as pd

import full_study as fs

ORDINAL_WEIGHTS_DIR = "experiments/full_runs/sensing_train"
ANGLE_WEIGHTS_DIR = "experiments/full_runs_angle_tagged/sensing_train"
OUT_DIR = "experiments/full_runs/summary"
SEEDS = (0, 1, 2)
EVAL_EPISODES = 15
MAX_STEPS = 350
NOMINAL = dict(beam_count=3, fov_deg=90.0, noise_sigma=0.0, dropout_p=0.0)


class ShiftedSensingEnv(fs.SensingGameEnv):
    """SensingGameEnv variant supporting a beam-order permutation and/or a
    constant angular placement shift, for the stress tests above. Reuses
    the parent class's physics/reward/checkpoint logic verbatim; only
    _sensor_reading is overridden."""

    def __init__(self, *args, permute=False, angle_shift_deg=0.0, **kwargs):
        self.permute = permute
        self.angle_shift_deg = angle_shift_deg
        super().__init__(*args, **kwargs)

    def _sensor_reading(self):
        cfg = self.sensing
        base_angles = fs.beam_angles_for(cfg["beam_count"], cfg["fov_deg"])
        angles = base_angles + self.angle_shift_deg
        raw = fs.cast_rays_vectorized(self.car.x, self.car.y, self.car.angle,
                                       self.track_mask, angles)
        if cfg["noise_sigma"] > 0:
            raw = raw + np.random.normal(0, cfg["noise_sigma"] * fs.MAX_RANGE, size=raw.shape)
        if cfg["dropout_p"] > 0:
            drop_mask = np.random.rand(len(raw)) < cfg["dropout_p"]
            raw = np.where(drop_mask, fs.MAX_RANGE, raw)
        raw = np.clip(raw, 0, fs.MAX_RANGE)
        norm = raw / fs.MAX_RANGE
        d_left, d_right = norm[0], norm[-1]
        curvature = float(np.clip(abs(d_left - d_right) / max(d_left + d_right, 1.0), 0.0, 1.0))

        norm_enc, angles_enc = norm, angles
        if self.permute and len(norm) > 1:
            # Deliberate, guaranteed-non-identity permutation (a reversal),
            # rather than a random seeded one: a random permutation of a
            # short array can coincide with the identity by chance (e.g.
            # RandomState(1234).permutation(3) == [0,1,2]), which would
            # silently make this "permutation" test a no-op exactly at the
            # 3-beam reference condition this experiment cares about most.
            perm = np.arange(len(norm))[::-1]
            norm_enc = norm[perm]
            angles_enc = angles[perm]  # paired permutation: (angle, reading) stays self-consistent

        if self.encoding == "angle_tagged":
            encoded = fs.encode_angle_tagged(norm_enc, angles_enc)
        else:
            # ordinal: only norm order matters -- this is the vulnerability under test
            encoded = np.ones(fs.MAX_BEAMS, dtype=np.float32)
            encoded[: len(norm_enc)] = norm_enc
        return encoded, float(d_left), float(d_right), curvature


def eval_condition(weights_path, encoding, state_size, seed, permute, angle_shift_deg,
                    episodes=EVAL_EPISODES, max_steps=MAX_STEPS):
    fs.set_global_seed(seed + 9000)
    env = ShiftedSensingEnv(max_steps=max_steps, sensing=NOMINAL, encoding=encoding,
                             permute=permute, angle_shift_deg=angle_shift_deg)
    cfg = fs.ABLATION_CONFIGS["d3qn_per"]
    agent = fs.Agent(state_size, env.action_size, **cfg)
    agent.model.load_weights(weights_path)
    rows = []
    for ep in range(episodes):
        state = env.reset()
        total_reward, steps = 0.0, 0
        done = False
        while not done:
            action = agent.act(state, greedy=True)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        rows.append(dict(episode=ep, total_reward=total_reward, steps=steps,
                          checkpoints=env.checkpoints_cleared, crashed=int(env.crashed)))
    return fs.summarize(pd.DataFrame(rows))


def main():
    conditions = [
        ("baseline", False, 0.0),
        ("beam_permutation", True, 0.0),
        ("angle_shift_15deg", False, 15.0),
        ("angle_shift_30deg", False, 30.0),
    ]
    rows = []
    for encoding, weights_dir, state_size in [
        ("ordinal", ORDINAL_WEIGHTS_DIR, fs.MAX_BEAMS + 2),
        ("angle_tagged", ANGLE_WEIGHTS_DIR, fs.N_CANONICAL + 2),
    ]:
        for seed in SEEDS:
            weights_path = os.path.join(weights_dir, f"nominal_seed{seed}", "final.weights.h5")
            if not os.path.exists(weights_path):
                print(f"WARNING: missing {weights_path}, skipping")
                continue
            for cond_name, permute, shift in conditions:
                s = eval_condition(weights_path, encoding, state_size, seed, permute, shift)
                s.update(encoding=encoding, seed=seed, condition=cond_name)
                rows.append(s)
                print(f"{encoding} seed={seed} {cond_name}: "
                      f"mean_checkpoints={s['mean_checkpoints']:.2f} "
                      f"crash_rate={s['crash_rate']:.2f}")

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "semantic_shift_stress_test.csv")
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}\n")

    summary = df.groupby(["encoding", "condition"])[["mean_checkpoints", "crash_rate"]].mean()
    print(summary.to_string())
    summary.to_csv(os.path.join(OUT_DIR, "semantic_shift_stress_test_summary.csv"))


if __name__ == "__main__":
    main()
