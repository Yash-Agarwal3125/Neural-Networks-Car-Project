"""
Full Phase-4 experimental study — designed to run on Google Colab (CPU or GPU
runtime; GPU gives little benefit here since the bottleneck is the per-step
Python/pygame simulation loop, not matrix multiplies, but a Colab CPU is
often faster than a laptop CPU and, more importantly, frees up the local
machine and lets several phases run in parallel across separate sessions).

Delivers exactly the two "central empirical objects" already promised in
`latex_source/paper.tex` Sections 1-3, at a scope that is actually finishable:

  PHASE A — Component-wise ablation matrix (Sec. 3 "Benchmarking Protocol", item 2)
    {DQN, Double DQN, Dueling DQN, D3QN} x {no PER, PER}  = 8 configs x 3 seeds

  PHASE B — Sensing-sweep grid (Sec. 3 "Benchmarking Protocol", item 3)
    One-factor-at-a-time sensitivity design around the nominal 3-beam/90 deg/
    noise=0/dropout=0 operating point (this concretizes the "..." placeholders
    left in the original text for noise/dropout levels):
      beam_count in {1, 3, 5, 7}         (fov=90, noise=0,   dropout=0)
      fov_deg    in {60, 90, 120, 150}   (beams=3, noise=0,   dropout=0)
      noise_sigma in {0, 0.02, 0.05, 0.1}(beams=3, fov=90,    dropout=0)
      dropout_p  in {0, 0.1, 0.2, 0.4}   (beams=3, fov=90,    noise=0)
    Two policies are trained (3 seeds each): NOMINAL (always the fixed
    3-beam/90/0/0 point) and RANDOMIZED (grid point resampled every episode).
    Both are then evaluated, under the same headless harness, at every grid
    point above -> "evaluated robustness" (nominal-trained) vs "trained
    robustness" (randomized-trained).

Key engineering fix vs. the original repo/`ablation.py`: ray casting is
vectorized with numpy over (beams x range-steps) instead of a Python loop
calling `pygame.Surface.get_at()` per pixel, which is the dominant per-step
cost. This is what makes Phase B computationally tractable at all.

Usage (each phase is independent and safe to run in a separate Colab session
in parallel; everything writes incrementally to --out_dir so partial results
survive a Colab disconnect):

    python full_study.py --phase ablation      --config d3qn_per --seed 0
    python full_study.py --phase sensing_train --regime nominal    --seed 0
    python full_study.py --phase sensing_train --regime randomized --seed 0
    python full_study.py --phase sensing_eval  --regime nominal    --seed 0
    python full_study.py --phase driver   # runs EVERYTHING sequentially
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYTHONHASHSEED"] = "0"

import argparse
import itertools
import json
import random
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import pygame
import tensorflow as tf
from keras import Model, layers, optimizers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Core_Game_Parts import (
    Car, checkpoint_data, DRAW_COLOR,
    SCREEN_WIDTH, SCREEN_HEIGHT, TRACK_IMAGE_PATH, CAR_IMAGE_PATH,
)

MAX_BEAMS = 7
MAX_RANGE = 200
NOMINAL = dict(beam_count=3, fov_deg=90.0, noise_sigma=0.0, dropout_p=0.0)

# ---- Phase A: 8-cell algorithmic ablation ---------------------------- #
ABLATION_CONFIGS = {
    "dqn":         dict(dueling=False, double=False, per=False),
    "dqn_per":     dict(dueling=False, double=False, per=True),
    "ddqn":        dict(dueling=False, double=True,  per=False),
    "ddqn_per":    dict(dueling=False, double=True,  per=True),
    "dueling":     dict(dueling=True,  double=False, per=False),
    "dueling_per": dict(dueling=True,  double=False, per=True),
    "d3qn":        dict(dueling=True,  double=True,  per=False),
    "d3qn_per":    dict(dueling=True,  double=True,  per=True),
}

# ---- Phase B: one-factor-at-a-time sensing grid ----------------------- #
SENSING_GRID = (
    [dict(NOMINAL, beam_count=b) for b in (1, 3, 5, 7)]
    + [dict(NOMINAL, fov_deg=f) for f in (60, 90, 120, 150)]
    + [dict(NOMINAL, noise_sigma=s) for s in (0.0, 0.02, 0.05, 0.1)]
    + [dict(NOMINAL, dropout_p=p) for p in (0.0, 0.1, 0.2, 0.4)]
)


def dedup_grid(grid):
    seen, out = set(), []
    for g in grid:
        key = tuple(sorted(g.items()))
        if key not in seen:
            seen.add(key)
            out.append(g)
    return out


SENSING_GRID = dedup_grid(SENSING_GRID)  # 13 distinct points (nominal shared)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------- #
# Vectorized ray casting: (n_beams x MAX_RANGE) numpy op instead of a
# nested Python loop over pygame.Surface.get_at(). This is the change that
# makes the sensing sweep affordable.
# --------------------------------------------------------------------- #
def cast_rays_vectorized(car_x, car_y, car_angle, track_mask, beam_angles_deg, max_range=MAX_RANGE):
    angles = np.radians(car_angle + np.asarray(beam_angles_deg, dtype=np.float64))
    steps = np.arange(1, max_range + 1, dtype=np.float64)
    xs = car_x + np.cos(angles)[:, None] * steps[None, :]
    ys = car_y - np.sin(angles)[:, None] * steps[None, :]
    xi = xs.astype(np.int32)
    yi = ys.astype(np.int32)
    w, h = track_mask.shape
    inbounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xi_c = np.clip(xi, 0, w - 1)
    yi_c = np.clip(yi, 0, h - 1)
    hit = track_mask[xi_c, yi_c] & inbounds
    stop = hit | (~inbounds)
    any_stop = stop.any(axis=1)
    first_idx = np.argmax(stop, axis=1)
    distances = np.where(any_stop, first_idx + 1, max_range).astype(np.float64)
    return distances


def beam_angles_for(beam_count, fov_deg):
    if beam_count <= 1:
        return np.array([0.0])
    return np.linspace(-fov_deg / 2.0, fov_deg / 2.0, beam_count)


# --------------------------------------------------------------------- #
# Angle-invariant (canonical-angle-grid) sensing encoding.
#
# The original state vector pads beam readings into a fixed-length array by
# ORDINAL POSITION (padded[:len(norm)] = norm): array index carries no
# physical-angle meaning, so a network trained at beam_count=3 has no basis
# for interpreting a beam_count=5 reading placed at the same array index
# (see the beam-count brittleness result in Section IV-B). This encoding
# instead reserves one array slot per canonical physical angle -- spanning
# the union of all swept FOVs at a resolution (5 deg) finer than the
# smallest actual inter-beam spacing across the sensing grid (10 deg, at
# beam_count=7/fov=60) -- so array index means the same physical direction
# for every (beam_count, fov_deg) configuration. Unfilled bins (no beam
# placed there for this configuration) get the same max-range sentinel used
# for the ordinal encoding's padding.
# --------------------------------------------------------------------- #
CANONICAL_ANGLES = np.linspace(-75.0, 75.0, 31)  # 5-degree resolution
N_CANONICAL = len(CANONICAL_ANGLES)


def encode_angle_tagged(norm_readings, beam_angles_deg):
    encoded = np.ones(N_CANONICAL, dtype=np.float32)  # sentinel = max range
    beam_angles_deg = np.asarray(beam_angles_deg, dtype=np.float64)
    nearest_bin = np.argmin(
        np.abs(CANONICAL_ANGLES[None, :] - beam_angles_deg[:, None]), axis=1)
    encoded[nearest_bin] = norm_readings
    return encoded


# --------------------------------------------------------------------- #
# Environment: same reward/physics as experiments/ablation.py (which fixed
# MANUAL.md's dead reward_breakdown bug), generalized to configurable
# sensing. Nominal-sensing use (beam_count=3, fov=90, noise=0, dropout=0)
# reduces exactly to the original 3-ray sensor model.
# --------------------------------------------------------------------- #
class SensingGameEnv:
    def __init__(self, max_steps=350, sensing=None, randomize_sensing=False,
                 sensing_choices=None, encoding="ordinal"):
        pygame.init()
        pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        surf = pygame.image.load(TRACK_IMAGE_PATH).convert()
        arr = pygame.surfarray.array3d(surf)  # (W, H, 3)
        self.track_mask = np.all(arr == np.array(DRAW_COLOR, dtype=arr.dtype), axis=2)
        assert encoding in ("ordinal", "angle_tagged")
        self.encoding = encoding
        # beams (padded, ordinal position OR canonical angle bins) + speed + curvature
        self.state_size = (N_CANONICAL if encoding == "angle_tagged" else MAX_BEAMS) + 2
        self.action_size = 4
        self.max_steps = max_steps
        self.randomize_sensing = randomize_sensing
        self.sensing_choices = sensing_choices or SENSING_GRID
        self.sensing = dict(sensing or NOMINAL)
        self.reset()

    def _sample_sensing(self):
        if self.randomize_sensing:
            self.sensing = dict(random.choice(self.sensing_choices))

    def reset(self):
        self._sample_sensing()
        self.car = Car(CAR_IMAGE_PATH, 900, 426, angle=-45)
        self.steps = 0
        self.checkpoints_cleared = 0
        self.current_checkpoint_idx = 0
        self.inside_checkpoint = False
        self.prev_dist_to_next_checkpoint = None
        self.no_progress_steps = 0
        self.lap_start_time = time.time()
        self.lap_times = []
        self.crashed = False
        return self._get_state()

    def _sensor_reading(self):
        cfg = self.sensing
        angles = beam_angles_for(cfg["beam_count"], cfg["fov_deg"])
        raw = cast_rays_vectorized(self.car.x, self.car.y, self.car.angle,
                                    self.track_mask, angles)
        if cfg["noise_sigma"] > 0:
            raw = raw + np.random.normal(0, cfg["noise_sigma"] * MAX_RANGE, size=raw.shape)
        if cfg["dropout_p"] > 0:
            drop_mask = np.random.rand(len(raw)) < cfg["dropout_p"]
            raw = np.where(drop_mask, MAX_RANGE, raw)
        raw = np.clip(raw, 0, MAX_RANGE)
        norm = raw / MAX_RANGE
        d_left, d_right = norm[0], norm[-1]
        curvature = float(np.clip(abs(d_left - d_right) / max(d_left + d_right, 1.0), 0.0, 1.0))
        if self.encoding == "angle_tagged":
            encoded = encode_angle_tagged(norm, angles)
        else:
            encoded = np.ones(MAX_BEAMS, dtype=np.float32)
            encoded[: len(norm)] = norm
        return encoded, float(d_left), float(d_right), curvature

    def step(self, action):
        done = False
        reward = 0.0
        MAX_SPEED, SAFE_TURN_SPEED, MIN_SPEED = 10, 4.5, 1.5

        if action == 0:
            self.car.angle += 5
        elif action == 2:
            self.car.angle -= 5
        if action == 1:
            self.car.speed = min(self.car.speed + 0.15, MAX_SPEED)
        elif action == 3:
            self.car.speed = max(self.car.speed - 0.30, MIN_SPEED)
        else:
            self.car.speed = max(self.car.speed - 0.12, 2.0)
        self.car.move()
        self.steps += 1

        beams, left, right, curvature = self._sensor_reading()

        target_rect = checkpoint_data[self.current_checkpoint_idx]
        target_x = target_rect[0] + target_rect[2] / 2
        target_y = target_rect[1] + target_rect[3] / 2
        curr_dist = np.hypot(self.car.x - target_x, self.car.y - target_y)
        progress = 0.0 if self.prev_dist_to_next_checkpoint is None else self.prev_dist_to_next_checkpoint - curr_dist
        self.prev_dist_to_next_checkpoint = curr_dist
        r_progress = float(np.clip(progress, -1.0, 1.0))

        self.no_progress_steps = 0 if r_progress >= 0.2 else self.no_progress_steps + 1
        if self.no_progress_steps > 250:
            reward -= 2.0
            done = True

        desired_speed = np.clip((1.0 - curvature) * MAX_SPEED + curvature * SAFE_TURN_SPEED, MIN_SPEED, MAX_SPEED)
        speed_error = self.car.speed - desired_speed
        r_speed = float(np.exp(-0.5 * (speed_error ** 2)))
        if curvature > 0.4 and action != 3:
            reward -= 0.15
        if curvature < 0.1 and self.car.speed < 6.0:
            reward -= 0.01
        if curvature < 0.2:
            reward += 1.2 * (self.car.speed / MAX_SPEED)
        if action == 3:
            reward -= 0.4 * (1.0 - curvature) * (self.car.speed / MAX_SPEED)

        r_center = float(np.clip(1.0 - abs(left - right), 0.0, 1.0))

        if action in (0, 2) and self.car.speed > 4.5:
            reward -= 0.4

        checkpoint_hit = False
        cp = checkpoint_data[self.current_checkpoint_idx]
        cp_rect = pygame.Rect(cp[0], cp[1], cp[2], cp[3])
        cp_rect.inflate_ip(40, 40)
        inside_now = self.car.get_rect().colliderect(cp_rect)
        if inside_now and not self.inside_checkpoint:
            checkpoint_hit = True
            self.inside_checkpoint = True
        if not inside_now:
            self.inside_checkpoint = False

        lap_time = None
        r_step = -0.03
        if checkpoint_hit:
            if self.current_checkpoint_idx < len(checkpoint_data) - 1:
                reward += 20.0
            self.no_progress_steps = 0
            self.checkpoints_cleared += 1
            self.current_checkpoint_idx += 1
            self.prev_dist_to_next_checkpoint = None

        if self.current_checkpoint_idx >= len(checkpoint_data):
            self.current_checkpoint_idx = 0
            lap_time = time.time() - self.lap_start_time
            if not hasattr(self, "best_lap_time"):
                self.best_lap_time = lap_time
                reward += 50
            else:
                improvement = self.best_lap_time - lap_time
                if improvement > 0:
                    reward += 200 * improvement
                    self.best_lap_time = lap_time
                else:
                    reward -= 30
            self.lap_times.append(lap_time)
            self.lap_start_time = time.time()

        x, y = int(self.car.x), int(self.car.y)
        crashed = False
        if x < 0 or y < 0 or x >= SCREEN_WIDTH or y >= SCREEN_HEIGHT:
            reward = -10.0
            done = True
            crashed = True
        else:
            if self.track_mask[min(x, SCREEN_WIDTH - 1), min(y, SCREEN_HEIGHT - 1)]:
                reward = -10.0
                done = True
                crashed = True
        self.crashed = crashed

        reward += 1.0 * r_progress + 2.0 * r_speed + 0.3 * r_center
        reward += r_step

        if self.steps >= self.max_steps:
            done = True

        info = {
            "checkpoints": self.checkpoints_cleared, "crashed": crashed, "lap_time": lap_time,
            "reward_breakdown": {"center": 0.3 * r_center, "speed": 2.0 * r_speed,
                                  "progress": 1.0 * r_progress, "step": r_step},
        }
        return self._get_state(), float(reward), done, info

    def _get_state(self):
        beams, left, right, curvature = self._sensor_reading()
        return np.concatenate([beams, [self.car.speed, curvature]]).astype(np.float32)


# --------------------------------------------------------------------- #
# Replay buffers / network / agent (identical to experiments/ablation.py)
# --------------------------------------------------------------------- #
class UniformMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience, td_error=None):
        self.buffer.append(experience)

    def sample(self, batch_size, beta=0.4):
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        return batch, idx, np.ones(batch_size, dtype=np.float32)

    def update_priorities(self, indices, td_errors):
        pass

    def __len__(self):
        return len(self.buffer)


class PERMemory:
    def __init__(self, capacity, alpha=0.5):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.epsilon = 1e-5

    def add(self, experience, td_error):
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return experiences, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = (abs(td_error) + self.epsilon) ** self.alpha

    def __len__(self):
        return len(self.buffer)


def build_network(input_shape, action_size, dueling):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    if dueling:
        value = layers.Dense(64, activation="relu")(x)
        value = layers.Dense(1, activation="linear")(value)
        advantage = layers.Dense(64, activation="relu")(x)
        advantage = layers.Dense(action_size, activation="linear")(advantage)
        q_values = layers.Lambda(
            lambda a: a[0] + (a[1] - tf.reduce_mean(a[1], axis=1, keepdims=True))
        )([value, advantage])
    else:
        y = layers.Dense(64, activation="relu")(x)
        q_values = layers.Dense(action_size, activation="linear")(y)
    model = Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=optimizers.Adam(learning_rate=2e-4, clipnorm=1.0), loss="mse")
    return model


class Agent:
    def __init__(self, state_size, action_size, dueling, double, per):
        self.state_size, self.action_size = state_size, action_size
        self.dueling, self.double, self.per = dueling, double, per
        self.gamma, self.batch_size, self.tau = 0.98, 64, 0.005
        self.memory = PERMemory(20000) if per else UniformMemory(20000)
        self.model = build_network((state_size,), action_size, dueling)
        self.target_model = build_network((state_size,), action_size, dueling)
        self.target_model.set_weights(self.model.get_weights())
        self.epsilon, self.epsilon_decay, self.epsilon_min = 1.0, 0.995, 0.1

    @staticmethod
    def _qvals(model, x):
        # Direct __call__ instead of model.predict(): predict() re-enters
        # Keras' full predict-loop machinery (retracing, callbacks) on every
        # invocation, which dominates wall time for many small (batch=1-2)
        # calls; a plain forward pass gives identical outputs for these
        # feedforward Dense/dueling nets with far less per-call overhead.
        return model(x, training=False).numpy()

    def act(self, state, greedy=False):
        if not greedy and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q = self._qvals(self.model, np.expand_dims(state, 0))
        return int(np.argmax(q[0]))

    def remember_warmup(self, state, action, reward, next_state, done):
        if self.per:
            self.memory.buffer.append((state, action, reward, next_state, done))
            self.memory.priorities.append(0.5)
        else:
            self.memory.add((state, action, reward, next_state, done))

    def remember(self, state, action, reward, next_state, done):
        if not self.per:
            self.memory.add((state, action, reward, next_state, done))
            return
        # state and next_state are batched into one online-network call
        # instead of two: identical outputs, one fewer model invocation.
        online_pair = self._qvals(self.model, np.stack([state, next_state]))
        q_values, best_next_online = online_pair[0], online_pair[1]
        target_q = self._qvals(self.target_model, np.expand_dims(next_state, 0))[0]
        if self.double:
            bootstrap = target_q[np.argmax(best_next_online)]
        else:
            bootstrap = np.max(target_q)
        target = reward + self.gamma * bootstrap * (1 - int(done))
        self.memory.add((state, action, reward, next_state, done), target - q_values[action])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states, next_states = np.array(states), np.array(next_states)
        # states and next_states batched into one online-network call.
        online_pair = self._qvals(self.model, np.concatenate([states, next_states], axis=0))
        targets, next_q_online = online_pair[:self.batch_size], online_pair[self.batch_size:]
        next_q_target = self._qvals(self.target_model, next_states)

        td_errors = []
        for i in range(self.batch_size):
            if self.double:
                bootstrap = next_q_target[i][np.argmax(next_q_online[i])]
            else:
                bootstrap = np.max(next_q_target[i])
            target_value = rewards[i] + self.gamma * bootstrap * (1 - dones[i])
            td_error = target_value - targets[i][actions[i]]
            td_errors.append(td_error)
            targets[i][actions[i]] += 0.1 * td_error

        self.model.fit(states, targets, sample_weight=weights, epochs=1, verbose=0)
        self.memory.update_priorities(indices, td_errors)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        new_weights = [self.tau * w + (1 - self.tau) * tw
                       for w, tw in zip(self.model.get_weights(), self.target_model.get_weights())]
        self.target_model.set_weights(new_weights)


# --------------------------------------------------------------------- #
# Training / evaluation drivers
# --------------------------------------------------------------------- #
def train_run(agent_kwargs, seed, episodes, max_steps, run_dir, env_kwargs=None, warmup_steps=500):
    set_global_seed(seed)
    os.makedirs(run_dir, exist_ok=True)
    env = SensingGameEnv(max_steps=max_steps, **(env_kwargs or {}))
    agent = Agent(env.state_size, env.action_size, **agent_kwargs)

    state = env.reset()
    for _ in range(warmup_steps):
        action = np.random.randint(agent.action_size)
        next_state, reward, done, _ = env.step(action)
        agent.remember_warmup(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()

    rows, t0 = [], time.time()
    for ep in range(episodes):
        state = env.reset()
        total_reward, steps, max_speed = 0.0, 0, 0.0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            steps += 1
            if len(agent.memory) > warmup_steps and steps % 4 == 0:
                agent.replay()
            state = next_state
            total_reward += reward
            max_speed = max(max_speed, env.car.speed)
        rows.append(dict(episode=ep, total_reward=total_reward, steps=steps,
                          checkpoints=env.checkpoints_cleared, crashed=int(env.crashed),
                          max_speed=max_speed, epsilon=agent.epsilon,
                          lap_time=env.lap_times[-1] if env.lap_times else None,
                          wall_time_s=time.time() - t0))
        if (ep + 1) % 20 == 0:
            print(f"[{run_dir}] ep {ep+1}/{episodes} reward={total_reward:.1f} "
                  f"cp={env.checkpoints_cleared} eps={agent.epsilon:.3f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(run_dir, "train_log.csv"), index=False)

    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "train_log.csv"), index=False)
    agent.model.save_weights(os.path.join(run_dir, "final.weights.h5"))
    meta = dict(seed=seed, episodes=episodes, max_steps=max_steps,
                env_kwargs=env_kwargs or {}, **agent_kwargs)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"DONE {run_dir} in {time.time()-t0:.0f}s")
    return run_dir


def eval_run(weights_path, agent_kwargs, seed, episodes, max_steps, sensing_point, state_size,
             action_size=4, encoding="ordinal"):
    set_global_seed(seed + 9000)
    env = SensingGameEnv(max_steps=max_steps, sensing=sensing_point, encoding=encoding)
    agent = Agent(state_size, action_size, **agent_kwargs)
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
                          checkpoints=env.checkpoints_cleared, crashed=int(env.crashed),
                          lap_completed=int(len(env.lap_times) > 0),
                          lap_time=env.lap_times[-1] if env.lap_times else None))
    return pd.DataFrame(rows)


def summarize(df, extra=None):
    s = dict(
        n_episodes=len(df), mean_reward=df.total_reward.mean(), std_reward=df.total_reward.std(),
        mean_checkpoints=df.checkpoints.mean(), checkpoint_clear_rate=(df.checkpoints >= 12).mean(),
        crash_rate=df.crashed.mean(), lap_complete_rate=df.lap_completed.mean() if "lap_completed" in df else None,
        mean_lap_time=df.loc[df.lap_time.notna(), "lap_time"].mean() if "lap_time" in df else None,
    )
    if extra:
        s.update(extra)
    return s


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True,
                    choices=["ablation", "sensing_train", "sensing_eval", "driver"])
    p.add_argument("--config", choices=list(ABLATION_CONFIGS.keys()))
    p.add_argument("--regime", choices=["nominal", "randomized"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--max_steps", type=int, default=350)
    p.add_argument("--eval_episodes", type=int, default=15)
    p.add_argument("--out_dir", default="experiments/full_runs")
    p.add_argument("--encoding", choices=["ordinal", "angle_tagged"], default="ordinal",
                    help="Sensing state encoding: 'ordinal' (paper baseline, array-position "
                         "padding) or 'angle_tagged' (canonical-angle-grid, beam-count-"
                         "invariant). Only affects sensing_train/sensing_eval/driver phases.")
    args = p.parse_args()

    if args.phase == "ablation":
        cfg = ABLATION_CONFIGS[args.config]
        train_run(cfg, args.seed, args.episodes, args.max_steps,
                   os.path.join(args.out_dir, "ablation", f"{args.config}_seed{args.seed}"))

    elif args.phase == "sensing_train":
        randomize = args.regime == "randomized"
        env_kwargs = dict(randomize_sensing=randomize) if randomize else dict(sensing=NOMINAL)
        env_kwargs["encoding"] = args.encoding
        cfg = ABLATION_CONFIGS["d3qn_per"]  # the paper's full architecture
        train_run(cfg, args.seed, args.episodes, args.max_steps,
                   os.path.join(args.out_dir, "sensing_train", f"{args.regime}_seed{args.seed}"),
                   env_kwargs=env_kwargs)

    elif args.phase == "sensing_eval":
        run_dir = os.path.join(args.out_dir, "sensing_train", f"{args.regime}_seed{args.seed}")
        weights_path = os.path.join(run_dir, "final.weights.h5")
        cfg = ABLATION_CONFIGS["d3qn_per"]
        state_size = (N_CANONICAL if args.encoding == "angle_tagged" else MAX_BEAMS) + 2
        out_rows = []
        for point in SENSING_GRID:
            df = eval_run(weights_path, cfg, args.seed, args.eval_episodes, args.max_steps,
                           point, state_size=state_size, encoding=args.encoding)
            s = summarize(df, extra=dict(regime=args.regime, seed=args.seed, **point))
            out_rows.append(s)
            print(json.dumps(s, default=str))
        out_dir = os.path.join(args.out_dir, "sensing_eval")
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(out_rows).to_csv(
            os.path.join(out_dir, f"{args.regime}_seed{args.seed}.csv"), index=False)

    elif args.phase == "driver":
        for cfg_name in ABLATION_CONFIGS:
            for seed in (0, 1, 2):
                cfg = ABLATION_CONFIGS[cfg_name]
                train_run(cfg, seed, args.episodes, args.max_steps,
                           os.path.join(args.out_dir, "ablation", f"{cfg_name}_seed{seed}"))
        state_size = (N_CANONICAL if args.encoding == "angle_tagged" else MAX_BEAMS) + 2
        for regime in ("nominal", "randomized"):
            for seed in (0, 1, 2):
                env_kwargs = dict(randomize_sensing=True) if regime == "randomized" else dict(sensing=NOMINAL)
                env_kwargs["encoding"] = args.encoding
                train_run(ABLATION_CONFIGS["d3qn_per"], seed, args.episodes, args.max_steps,
                           os.path.join(args.out_dir, "sensing_train", f"{regime}_seed{seed}"),
                           env_kwargs=env_kwargs)
        for regime in ("nominal", "randomized"):
            for seed in (0, 1, 2):
                run_dir = os.path.join(args.out_dir, "sensing_train", f"{regime}_seed{seed}")
                weights_path = os.path.join(run_dir, "final.weights.h5")
                out_rows = []
                for point in SENSING_GRID:
                    df = eval_run(weights_path, ABLATION_CONFIGS["d3qn_per"], seed,
                                   args.eval_episodes, args.max_steps, point, state_size=state_size,
                                   encoding=args.encoding)
                    out_rows.append(summarize(df, extra=dict(regime=regime, seed=seed, **point)))
                out_dir = os.path.join(args.out_dir, "sensing_eval")
                os.makedirs(out_dir, exist_ok=True)
                pd.DataFrame(out_rows).to_csv(os.path.join(out_dir, f"{regime}_seed{seed}.csv"), index=False)
        print("=== FULL STUDY DONE ===")


if __name__ == "__main__":
    main()
