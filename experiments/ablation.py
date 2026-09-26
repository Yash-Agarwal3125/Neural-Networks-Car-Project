"""
Component ablation study for the Phase-4 paper (Section 4).

Fixes applied relative to `old time code.py` / MANUAL.md Sec.8:
  1. reward_breakdown is now populated in info{} (dead-log bug fixed).
  2. Explicit seeding of random/numpy/tensorflow per run (reproducibility).
  3. A single parametrized agent covers all four ablation cells instead of
     hand-copied scripts, so only {double, dueling, per} differ between runs.
  4. A headless evaluation pass (epsilon=0, no rendering) reports the
     checkpoint-clearance rate / crash rate / lap metrics MANUAL.md flagged
     as missing.

Four configs (isolating one component at a time, matching the paper's
Related-Work citations):
  DQN          : dueling=False, double=False, per=False   (plain baseline)
  Double DQN   : dueling=False, double=True,  per=False   (van Hasselt 2016)
  Dueling DQN  : dueling=True,  double=False, per=False   (Wang 2016)
  D3QN+PER     : dueling=True,  double=True,  per=True    (Schaul 2016 + ours)

Usage:
    python experiments/ablation.py --config dqn        --seed 0 --episodes 300
    python experiments/ablation.py --config double_dqn  --seed 0 --episodes 300
    python experiments/ablation.py --config dueling_dqn --seed 0 --episodes 300
    python experiments/ablation.py --config d3qn_per    --seed 0 --episodes 300
    python experiments/ablation.py --eval RUN_DIR --episodes 20
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYTHONHASHSEED"] = "0"

import argparse
import json
import random
import time
from collections import deque

import numpy as np
import pandas as pd
import pygame
import tensorflow as tf
from keras import Model, layers, optimizers

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Core_Game_Parts import (
    Car, ray_casting, checkpoint_data, DRAW_COLOR,
    SCREEN_WIDTH, SCREEN_HEIGHT, TRACK_IMAGE_PATH, CAR_IMAGE_PATH,
)

CONFIGS = {
    "dqn":          dict(dueling=False, double=False, per=False),
    "double_dqn":   dict(dueling=False, double=True,  per=False),
    "dueling_dqn":  dict(dueling=True,  double=False, per=False),
    "d3qn_per":     dict(dueling=True,  double=True,  per=True),
}


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------- #
# Environment (same physics/reward as old time code.py, with the dead
# reward_breakdown bug fixed so per-component curves are real).
# --------------------------------------------------------------------- #
class GameEnv:
    def __init__(self, max_steps=500):
        pygame.init()
        pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.track_surface = pygame.image.load(TRACK_IMAGE_PATH).convert()
        self.state_size = 5
        self.action_size = 4
        self.max_steps = max_steps
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.reset()

    def reset(self):
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

        sensor_distance, _ = ray_casting(self.car, self.track_surface)
        left, front, right = sensor_distance
        left, front, right = left / 200, front / 200, right / 200
        curvature = float(np.clip(abs(left - right) / max(left + right, 1.0), 0.0, 1.0))

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
            pixel = self.track_surface.get_at((x, y))[:3]
            if pixel == DRAW_COLOR:
                reward = -10.0
                done = True
                crashed = True
        self.crashed = crashed

        reward += 1.0 * r_progress + 2.0 * r_speed + 0.3 * r_center
        reward += r_step

        if self.steps >= self.max_steps:
            done = True

        info = {
            "checkpoints": self.checkpoints_cleared,
            "crashed": crashed,
            "lap_time": lap_time,
            "reward_breakdown": {
                "center": 0.3 * r_center,
                "speed": 2.0 * r_speed,
                "progress": 1.0 * r_progress,
                "step": r_step,
            },
        }
        return self._get_state(), float(reward), done, info

    def _get_state(self):
        sensor_distance, _ = ray_casting(self.car, self.track_surface)
        left, front, right = sensor_distance
        left, front, right = left / 200, front / 200, right / 200
        curvature = abs(left - right) / max(left + right, 1.0)
        return np.array([left, front, right, self.car.speed, curvature], dtype=np.float32)


# --------------------------------------------------------------------- #
# Replay buffers
# --------------------------------------------------------------------- #
class UniformMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience, td_error=None):
        self.buffer.append(experience)

    def sample(self, batch_size, beta=0.4):
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        weights = np.ones(batch_size, dtype=np.float32)
        return batch, idx, weights

    def update_priorities(self, indices, td_errors):
        pass

    def __len__(self):
        return len(self.buffer)


class PERMemory:
    def __init__(self, capacity, alpha=0.5):
        self.capacity = capacity
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


# --------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------- #
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
        self.state_size = state_size
        self.action_size = action_size
        self.dueling = dueling
        self.double = double
        self.per = per
        self.gamma = 0.98
        self.batch_size = 64
        self.tau = 0.005
        self.memory = PERMemory(20000) if per else UniformMemory(20000)
        self.model = build_network((state_size,), action_size, dueling)
        self.target_model = build_network((state_size,), action_size, dueling)
        self.target_model.set_weights(self.model.get_weights())
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def act(self, state, greedy=False):
        if not greedy and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q = self.model.predict(np.expand_dims(state, 0), verbose=0)
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
        q_values = self.model.predict(np.expand_dims(state, 0), verbose=0)[0]
        target_q = self.target_model.predict(np.expand_dims(next_state, 0), verbose=0)[0]
        if self.double:
            best_next = np.argmax(self.model.predict(np.expand_dims(next_state, 0), verbose=0)[0])
            bootstrap = target_q[best_next]
        else:
            bootstrap = np.max(target_q)
        target = reward + self.gamma * bootstrap * (1 - int(done))
        td_error = target - q_values[action]
        self.memory.add((state, action, reward, next_state, done), td_error)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states, next_states = np.array(states), np.array(next_states)
        targets = self.model.predict(states, verbose=0)
        next_q_online = self.model.predict(next_states, verbose=0)
        next_q_target = self.target_model.predict(next_states, verbose=0)

        td_errors = []
        for i in range(self.batch_size):
            if self.double:
                best_action = np.argmax(next_q_online[i])
                bootstrap = next_q_target[i][best_action]
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

        new_weights = [
            self.tau * w + (1 - self.tau) * tw
            for w, tw in zip(self.model.get_weights(), self.target_model.get_weights())
        ]
        self.target_model.set_weights(new_weights)


# --------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------- #
def train(config_name, seed, episodes, max_steps, out_dir, warmup_steps=500):
    set_global_seed(seed)
    run_dir = os.path.join(out_dir, f"{config_name}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    env = GameEnv(max_steps=max_steps)
    cfg = CONFIGS[config_name]
    agent = Agent(env.state_size, env.action_size, **cfg)

    state = env.reset()
    for _ in range(warmup_steps):
        action = np.random.randint(agent.action_size)
        next_state, reward, done, _ = env.step(action)
        agent.remember_warmup(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()

    rows = []
    t0 = time.time()
    for ep in range(episodes):
        state = env.reset()
        total_reward, steps, max_speed = 0.0, 0, 0.0
        comp = {"center": 0.0, "speed": 0.0, "progress": 0.0, "step": 0.0}
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            steps += 1
            if len(agent.memory) > warmup_steps and steps % 4 == 0:
                agent.replay()
            for k, v in info["reward_breakdown"].items():
                comp[k] += v
            state = next_state
            total_reward += reward
            max_speed = max(max_speed, env.car.speed)

        rows.append({
            "episode": ep, "config": config_name, "seed": seed,
            "total_reward": total_reward, "steps": steps,
            "checkpoints": env.checkpoints_cleared, "crashed": int(env.crashed),
            "max_speed": max_speed, "epsilon": agent.epsilon,
            "lap_time": env.lap_times[-1] if env.lap_times else None,
            "r_center": comp["center"], "r_speed": comp["speed"],
            "r_progress": comp["progress"], "r_step": comp["step"],
            "wall_time_s": time.time() - t0,
        })
        if (ep + 1) % 25 == 0:
            print(f"[{config_name} seed{seed}] ep {ep+1}/{episodes} "
                  f"reward={total_reward:.1f} cp={env.checkpoints_cleared} "
                  f"eps={agent.epsilon:.3f} elapsed={time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(run_dir, "train_log.csv"), index=False)

    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "train_log.csv"), index=False)
    agent.model.save_weights(os.path.join(run_dir, "final.weights.h5"))
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({"config": config_name, "seed": seed, "episodes": episodes,
                    "max_steps": max_steps, **cfg}, f, indent=2)
    print(f"DONE {config_name} seed{seed} in {time.time()-t0:.0f}s -> {run_dir}")
    return run_dir


def evaluate(run_dir, episodes=20, max_steps=500):
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    set_global_seed(9999)
    env = GameEnv(max_steps=max_steps)
    agent = Agent(env.state_size, env.action_size,
                  dueling=meta["dueling"], double=meta["double"], per=meta["per"])
    agent.model.load_weights(os.path.join(run_dir, "final.weights.h5"))

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
        rows.append({
            "episode": ep, "total_reward": total_reward, "steps": steps,
            "checkpoints": env.checkpoints_cleared, "crashed": int(env.crashed),
            "lap_completed": int(len(env.lap_times) > 0),
            "lap_time": env.lap_times[-1] if env.lap_times else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "eval_log.csv"), index=False)
    summary = {
        "config": meta["config"], "seed": meta["seed"],
        "mean_reward": df.total_reward.mean(), "std_reward": df.total_reward.std(),
        "mean_checkpoints": df.checkpoints.mean(),
        "checkpoint_clear_rate": (df.checkpoints >= 12).mean(),
        "crash_rate": df.crashed.mean(),
        "lap_complete_rate": df.lap_completed.mean(),
        "mean_lap_time": df.loc[df.lap_time.notna(), "lap_time"].mean(),
    }
    with open(os.path.join(run_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS.keys()))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--out_dir", default="experiments/runs")
    p.add_argument("--eval", default=None, help="run_dir to evaluate instead of training")
    args = p.parse_args()

    if args.eval:
        evaluate(args.eval, episodes=args.episodes if args.episodes != 300 else 20,
                  max_steps=args.max_steps)
    else:
        train(args.config, args.seed, args.episodes, args.max_steps, args.out_dir)
