import numpy as np
import tensorflow as tf
from keras import Model, layers, optimizers
import random
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import pygame
import numpy as np
from Core_Game_Parts import *
import os 
import pandas as pda
def build_dueling_dqn(input_shape, action_size):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)

    # Value stream
    value = layers.Dense(64, activation='relu')(x)
    value = layers.Dense(1, activation='linear')(value)

    # Advantage stream
    advantage = layers.Dense(64, activation='relu')(x)
    advantage = layers.Dense(action_size, activation='linear')(advantage)

    # Combine value and advantage
    q_values = layers.Lambda(lambda a: a[0] + (a[1] - tf.reduce_mean(a[1], axis=1, keepdims=True)))([value, advantage])

    model = Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=optimizers.Adam(learning_rate=2e-4, clipnorm=1.0), loss='mse')

    return model
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
class D3QNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.98
        self.batch_size = 64
        self.tau = 0.005
        self.memory = PERMemory(20000)

        self.model = build_dueling_dqn((state_size,), action_size)
        self.target_model = build_dueling_dqn((state_size,), action_size)
        self.target_model.set_weights(self.model.get_weights())

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        q_values = self.model.predict(np.expand_dims(state, 0), verbose=0)[0]
        target_q = self.target_model.predict(np.expand_dims(next_state, 0), verbose=0)[0]
        best_next_action = np.argmax(self.model.predict(np.expand_dims(next_state, 0), verbose=0)[0])
        target = reward + self.gamma * target_q[best_next_action] * (1 - int(done))
        td_error = target - q_values[action]
        self.memory.add((state, action, reward, next_state, done), td_error)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        targets = self.model.predict(states, verbose=0)
        next_qs = self.model.predict(next_states, verbose=0)
        next_q_targets = self.target_model.predict(next_states, verbose=0)

        td_errors = []
        for i in range(self.batch_size):
            best_action = np.argmax(next_qs[i])
            target_value = rewards[i] + self.gamma * next_q_targets[i][best_action] * (1 - dones[i])
            td_error = target_value - targets[i][actions[i]]
            td_errors.append(td_error)
            targets[i][actions[i]] += 0.1 * td_error

        self.model.fit(states, targets, sample_weight=weights, epochs=1, verbose=0)
        self.memory.update_priorities(indices, td_errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        new_weights = []
        for w, target_w in zip(self.model.get_weights(), self.target_model.get_weights()):
            new_weights.append(self.tau * w + (1 - self.tau) * target_w)
        self.target_model.set_weights(new_weights)

    def remember_warmup(self, state, action, reward, next_state, done):
    # No NN calls, no TD-error
        self.memory.buffer.append((state, action, reward, next_state, done))
        self.memory.priorities.append(0.5)


class GameEnv:
    def __init__(self, render_mode=False):
        # Disable rendering (headless)
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        
        self.render_mode = render_mode
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.track_surface = pygame.image.load(TRACK_IMAGE_PATH).convert()
        self.car = Car(CAR_IMAGE_PATH, 900, 426, angle=-45)
        self.clock = pygame.time.Clock()
        
        self.state_size = 5   # 3 sensors + speed + curvature
        self.action_size = 4  # left, right, brake, accelerate
        self.max_steps = 1000
        self.checkpoints_cleared = 0
        self.prev_dist_to_next_checkpoint = None
        self.current_checkpoint_idx = 0

        self.current_checkpoint_idx = 0
        self.checkpoints_cleared = 0
        self.inside_checkpoint = False  
        self.no_progress_steps = 0
        self.curriculum_max_checkpoint = 1   # start with only checkpoint 0 → 1
        self.total_checkpoints = len(checkpoint_data)

        if self.render_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    def reset(self):
        self.current_checkpoint_idx = 0
        self.checkpoints_cleared = 0
        self.inside_checkpoint = False
        self.prev_dist_to_next_checkpoint = None

        self.car = Car(CAR_IMAGE_PATH, 900, 426, angle=-45)
        self.steps = 0
        self.checkpoints_cleared = 0
        self.prev_dist_to_next_checkpoint = None
        self.current_checkpoint_idx = 0
        self.lap_start_time = time.time()
        self.lap_times = []
        return self._get_state()
   
    def step(self, action):
        done = False
        reward = 0.0

    # =====================================================
    # 1. PHYSICS UPDATE (ACCEL / BRAKE / TURN)
    # =====================================================
        MAX_SPEED = 10
        SAFE_TURN_SPEED = 4.5 
        MIN_SPEED = 1.5

    # Steering
        if action == 0:      # left
            self.car.angle += 5
        elif action == 2:    # right
            self.car.angle -= 5

    # Speed control
        if action == 1:      # accelerate
            self.car.speed = min(self.car.speed + 0.15, MAX_SPEED)
        elif action == 3:    # brake
            self.car.speed = max(self.car.speed - 0.30, MIN_SPEED)
        else:                # turning friction
            self.car.speed = max(self.car.speed - 0.12, 2.0)
        self.car.move()
        self.steps += 1

    # =====================================================
    # 2. SENSOR READINGS
    # =====================================================
        sensor_distance, _ = ray_casting(self.car, self.track_surface)
        left, front, right = sensor_distance
        left/=200
        front/=200
        right/=200
        curvature = abs(left - right) / max(left + right, 1.0)
        curvature = np.clip(curvature, 0.0, 1.0)

    # =====================================================
    # 3. PROGRESS REWARD (MAIN OBJECTIVE)
    # =====================================================
        target_rect = checkpoint_data[self.current_checkpoint_idx]
        target_x = target_rect[0] + target_rect[2] / 2
        target_y = target_rect[1] + target_rect[3] / 2
        curr_dist = np.hypot(self.car.x - target_x, self.car.y - target_y)

        if self.prev_dist_to_next_checkpoint is None:
            progress = 0.0
        else:
            progress = self.prev_dist_to_next_checkpoint - curr_dist

        self.prev_dist_to_next_checkpoint = curr_dist
        r_progress = np.clip(progress, -1.0, 1.0)

        if r_progress < 0.2:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        if self.no_progress_steps > 250:
            reward -= 2.0
            done = True
    # =====================================================
    # 4. TURN-AWARE SPEED REWARD (KEY FIX)
    # =====================================================

        # Desired speed depends on curvature
        # Straight → fast, Turn → slow
        desired_speed = (
            (1.0 - curvature) * MAX_SPEED + curvature * SAFE_TURN_SPEED
            )
        desired_speed = np.clip(desired_speed, MIN_SPEED, MAX_SPEED)
        speed_error = self.car.speed - desired_speed

        # Reward matching desired speed
        r_speed = np.exp(-0.5 * (speed_error ** 2))
        if curvature > 0.4 and action != 3:
            reward -= 0.15

        if curvature < 0.1 and self.car.speed < 6.0:
            reward -= 0.01

        if curvature < 0.2:  # straight or gentle curve
            reward += 1.2 * (self.car.speed / MAX_SPEED)

        # Smooth braking penalty
        if action == 3:
            brake_penalty = (1.0 - curvature) * (self.car.speed / MAX_SPEED)
            reward -= 0.4 * brake_penalty



    # =====================================================
    # 5. CENTERING REWARD (STABILITY)
    # =====================================================
        r_center = 1.0 - abs(left - right)
        r_center = np.clip(r_center, 0.0, 1.0)


    # =====================================================
    # 6. ANTI-BAD BEHAVIOR PENALTIES
    # =====================================================
    # Penalize fast turning
        if action in [0, 2] and self.car.speed > 4.5:
            reward -= 0.4

    # Small step penalty (encourage efficiency)

        # Enforce curriculum boundary
        if self.current_checkpoint_idx >= self.curriculum_max_checkpoint and r_progress > 0:
            # Do NOT allow further progress
            reward -= 0.1

    # =====================================================
        # CURRICULUM-AWARE CHECKPOINT HANDLING
        # =====================================================
        # ---------- CHECKPOINT LOGIC (FOOL-PROOF) ----------
        checkpoint_hit = False

        cp = checkpoint_data[self.current_checkpoint_idx]
        cp_rect = pygame.Rect(cp[0], cp[1], cp[2], cp[3])
        cp_rect.inflate_ip(40, 40)

        car_rect = self.car.get_rect()

        inside_now = car_rect.colliderect(cp_rect)

        # Edge trigger: only on ENTER
        if inside_now and not self.inside_checkpoint:
            checkpoint_hit = True
            self.inside_checkpoint = True

        # Reset latch when fully outside
        if not inside_now:
            self.inside_checkpoint = False

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
                            # penalize slower laps


                self.lap_times.append(lap_time)
                self.lap_start_time = time.time()    







    # =====================================================
    # 8. COLLISION CHECK
    # =====================================================
        x, y = int(self.car.x), int(self.car.y)
        if x < 0 or y < 0 or x >= SCREEN_WIDTH or y >= SCREEN_HEIGHT:
            reward = -10.0
            done = True
        else:
            pixel = self.track_surface.get_at((x, y))[:3]
            if pixel == DRAW_COLOR:
                reward = -10.0
                done = True

    # =====================================================
    # 9. FINAL REWARD COMPOSITION
    # =====================================================
        reward += (
            1.0 * r_progress +
            2.0 * r_speed +
            0.3 * r_center
        )
        reward -= 0.03   # time penalty per step

    # =====================================================
    # 10. TERMINATION
    # =====================================================
        if self.steps >= self.max_steps:
            done = True

        state = self._get_state()
        info = {"checkpoints": self.checkpoints_cleared}

        return state, float(reward), done, info


    def _get_state(self):  
        sensor_distance, _ = ray_casting(self.car, self.track_surface)
        left, front, right = sensor_distance
        left/=200
        front/=200
        right/=200
        curvature = abs(left - right) / max(left + right, 1.0)
        return np.array([left, front, right, self.car.speed, curvature], dtype=np.float32)


    def render(self):
        if not self.render_mode:
            return
        self.screen.blit(self.track_surface, (0, 0))
        self.car.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

def safe_policy_detected(lap_times, max_speeds, window=30, tol=0.5):
    recent_laps = [t for t in lap_times[-window:] if t is not None]
    if len(recent_laps) < window // 2:
        return False

    lap_std = np.std(recent_laps)
    speed_mean = np.mean(max_speeds[-window:])

    if lap_std < tol and speed_mean > 0.9 * 10.0:
        return True
    return False


def train_agent(env, agent, episodes=500,  save_path='d3qn.weights.h5'):
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    plt.style.use('seaborn-v0_8-darkgrid')

    PHASE_1 = 1   # Basic driving & survival
    PHASE_2 = 2   # Exploration & speed
    PHASE_3 = 3   # safe lap (current)
    PHASE_4 = 4   # faster laps
    PHASE_5 = 5   # aggressive racing

    training_phase = PHASE_3 # Lap completion & smoothness
    def maybe_switch_phase(training_phase, lap_time_history, max_speed_history, agent):
        # ==========================
        # PHASE 3 → PHASE 4
        # ==========================
        if training_phase == 3:
            recent_laps = [t for t in lap_time_history[-40:] if t is not None]

            # REQUIRE:
            # 1. At least 20 completed laps
            # 2. Clear downward trend
            # 3. Stable high speed
            if len(recent_laps) >= 20:
                lap_std = np.std(recent_laps)
                lap_trend = recent_laps[0] - recent_laps[-1]
                speed_mean = np.mean(max_speed_history[-40:])

                if lap_std < 2.0 and lap_trend > 5.0 and speed_mean > 8.5:
                    print("\nSwitching to PHASE 4 (Lap Time Optimization)")

                    # LOCK exploration
                    agent.epsilon = 0.02
                    agent.epsilon_min = 0.02
                    agent.epsilon_decay = 1.0

                    # VERY IMPORTANT
                    agent.memory.buffer.clear()
                    agent.memory.priorities.clear()
                    print("Replay buffer cleared (racing phase)")

                    return 4

        return training_phase

    # History trackers
    scores_history = []
    avg_rewards = []
    loss_history = []
    max_speed_history = []
    checkpoints_history = []
    log_data = []
    step_count=[]
    time_per_episode=[]
    FPS_history=[]
    lap_time_history = []
    best_avg_reward = -float("inf")
    no_improve_episodes = 0
    plateau_patience = 50  # how long to wait before we “unstick” the agent
    WARMUP_STEPS = 3000
    state = env.reset()
    TOP_K = 3
    top_k_models = []  
    # ---- FAST & SAFE WARM-UP ----
    for _ in range(1000):
        action = np.random.randint(agent.action_size)
        next_state, reward, done, _ = env.step(action)
        agent.remember_warmup(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()

    # Optional dynamic reward tweak (progress boost)
    # progress_boost = 1.0
    best_lap_time = float("inf")
    best_model_path = "best_laptime.weights.h5"

    print("Starting training...")
    for episode in range(episodes):
        current_time_start=time.time()
        state = env.reset()
        total_reward = 0
        done = False
        episode_loss = []
        episode_max_speed = 0
        episode_checkpoints = getattr(env, "checkpoints_cleared", 0) if hasattr(env, "checkpoints_cleared") else 0
        total_steps = 0
        step=0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            if "checkpoints" in info:
                episode_checkpoints = info["checkpoints"]
            else:
                episode_checkpoints = getattr(env, "checkpoints_cleared", episode_checkpoints)
            reward_breakdown = info.get('reward_breakdown', {})
            log_data.append({
                'episode': episode,
                'step': step,
                'total_reward': reward,
                'r_center': reward_breakdown.get('center', 0),
                'r_speed': reward_breakdown.get('speed', 0),
                'r_progress': reward_breakdown.get('progress', 0),
                'r_step': reward_breakdown.get('step', 0),
                'checkpoints': info.get('checkpoints', 0)
            })
            agent.remember(state, action, reward, next_state, done)
            total_steps += 1
            step += 1
            if len(agent.memory) > WARMUP_STEPS and total_steps % 8 == 0:
                agent.replay()
            # Calculate loss proxy (difference between weights)
            # if len(agent.memory) > agent.batch_size:
            #     prev_weights = agent.model.get_weights()
            #     new_weights = agent.model.get_weights()
            #     episode_loss.append(np.mean([np.mean(np.abs(n - p)) for n, p in zip(new_weights, prev_weights)]))

            # Track metrics
            state = next_state
            total_reward += reward

            episode_max_speed = max(episode_max_speed, getattr(env.car, "speed", 0))
            if hasattr(env, "checkpoints_cleared"):
                episode_checkpoints = env.checkpoints_cleared

            current_time_end=time.time()-current_time_start
            if training_phase == PHASE_1:
                if total_steps > 600 and episode_checkpoints == 0:
                    reward -= 2.0
                    break

            elif training_phase == PHASE_2:
                if total_steps > 800 and episode_checkpoints == 0:
                    reward -= 1.0
                    break

            elif training_phase == PHASE_3:
                pass  # NO early termination


        # Store histories
        step_count.append(total_steps)
        scores_history.append(total_reward)
        loss_history.append(np.mean(episode_loss) if episode_loss else 0)
        max_speed_history.append(episode_max_speed)
        checkpoints_history.append(episode_checkpoints)
        avg_rewards.append(np.mean(scores_history[-50:]))
        time_per_episode.append(current_time_end)
        FPS_history.append(total_steps / max(time_per_episode[-1], 1e-6))

        # --- Plateau detection logic ---
        current_avg = avg_rewards[-1]
        if current_avg > best_avg_reward:
            best_avg_reward = current_avg
            no_improve_episodes = 0
        else:
            no_improve_episodes += 1

        # Freeze exploration once laps are consistent
        agent.epsilon = max(agent.epsilon, agent.epsilon_min)


        # After episode ends
        if len(env.lap_times) > 0:
            lap_time = env.lap_times[-1]
        else:
            lap_time = None

        lap_time_history.append(lap_time)


        # Update graphs every 5 episodes
        if (episode + 1) % 5 == 0:
            clear_output(wait=True)
            fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(12, 22))

            # --- Graph 1: Score ---
            ax1.set_title('Agent Score Over Time')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward (Score)')
            ax1.plot(scores_history, label='Score per Episode', color='royalblue')
            ax1.plot(avg_rewards, label='50-Episode Average', color='orange', linestyle='--')
            ax1.legend()

            # --- Graph 2: Max Speed ---
            ax2.set_title('Max Speed Achieved per Episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Max Speed')
            ax2.plot(max_speed_history, label='Max Speed', color='purple')
            ax2.legend()

            # --- Graph 3: Model Loss ---
            ax3.set_title('Episode Steps')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Steps per episode')
            ax3.plot(step_count, label='Steps per Episode', color='green', alpha=0.7)
            ax3.legend()

            # --- Graph 4: Checkpoints Cleared ---
            ax4.set_title('Checkpoints Cleared per Episode')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Checkpoints Cleared')
            episodes_range = range(len(checkpoints_history))
            ax4.bar(episodes_range, checkpoints_history, color='forestgreen', label='Checkpoints')
            ax4.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax4.legend()

            # --- Graph 5: Time per Episode ---
            ax5.set_title('Time Taken per Episode')
            ax5.set_xlabel('Episodes')
            ax5.set_ylabel('Time(Seconds)')
            ax5.plot(time_per_episode, label='Time per Episode', color='goldenrod')
            ax5.legend()

            # --- Graph 6: Lap Time ---
            ax6.set_title("Lap Time vs Episode")
            ax6.set_xlabel("Episode")
            ax6.set_ylabel("Lap Time (seconds)")

            valid_laps = [(i, t) for i, t in enumerate(lap_time_history) if t is not None]
            if valid_laps:
                ep, times = zip(*valid_laps)
                ax6.plot(ep, times, marker='o', alpha=0.7)
            ax6.legend()   
            plt.grid(True)
            plt.show()

            # Histogram
            plt.figure(figsize=(8, 6))
            plt.title('Distribution of Checkpoints Cleared')
            plt.xlabel('Checkpoints Cleared')
            plt.ylabel('Number of Episodes')
            plt.hist(checkpoints_history, bins=range(max(checkpoints_history) + 2), align='left', rwidth=0.8)
            plt.grid(axis='y', alpha=0.75)
            plt.show()

        print(f"Episode {episode+1}/{episodes} | Reward: {total_reward:.2f} | Avg: {avg_rewards[-1]:.2f} | Time: {time_per_episode[-1]:.2f} | Max Speed: {episode_max_speed:.2f} | Epsilon: {agent.epsilon:.3f} | FPS: {FPS_history[-1]:.2f}")

        if (episode + 1) % 50 == 0:
            agent.model.save_weights(save_path)
            print(f"Weights saved at episode {episode+1}")
        if (episode + 1) % 200 == 0:
            agent.model.save_weights(f"old_{episode+1}.weights.h5")

        # if not hasattr(train_agent, "best_checkpoint_record"):
        #     train_agent.best_checkpoint_record = -1  # static variable to persist across episodes

        current_checkpoints = episode_checkpoints
        #uncomment for presentation 
        # if current_checkpoints > train_agent.best_checkpoint_record:
        #     train_agent.best_checkpoint_record = current_checkpoints
        #     best_file = f"best_d3qn_{current_checkpoints}checkpoints.weights.h5"
        #     agent.model.save_weights(best_file)
        #     print(f"New Record! Cleared {current_checkpoints} checkpoints — weights saved as {best_file}")
        pda.DataFrame(log_data).to_csv("reward_components_log.csv", index=False)

        # =========================================
        # TOP-K LAP TIME MODEL SAVING
        # =========================================
        EPS = 0.05  # seconds

        if lap_time is not None:
            # Check if lap is meaningfully new
            is_new = not any(abs(lap_time - t) < EPS for t in top_k_models)

            if is_new:
                candidate = top_k_models + [lap_time]
                candidate = sorted(candidate)[:TOP_K]

                # Only update if leaderboard changed
                if candidate != top_k_models:
                    top_k_models = candidate

                    print("Top lap times updated:")
                    for i, t in enumerate(top_k_models, start=1):
                        print(f"  {i}: {t:.2f}s")

                    # Rewrite ALL ranks so files always exist
                    for rank, t in enumerate(top_k_models, start=1):
                        filename = f"best_{rank}.weights.h5"
                        agent.model.save_weights(filename)

        # if episode_checkpoints >= env.curriculum_max_checkpoint:
        #     env.curriculum_max_checkpoint = min(
        #         env.curriculum_max_checkpoint + 1,
        #         env.total_checkpoints
        #     )
            # print(f"Curriculum expanded → now up to checkpoint {env.curriculum_max_checkpoint}")
        
        if safe_policy_detected(lap_time_history, max_speed_history):
            print("SAFE POLICY DETECTED — lap time not improving")


        training_phase = maybe_switch_phase(
            training_phase,
            lap_time_history,
            max_speed_history,
            agent
        )

    return scores_history, avg_rewards, loss_history, max_speed_history, checkpoints_history

def test_agent(env, agent, episodes=10, render=True):
    print("\nStarting Evaluation Phase...\n")
    total_scores = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = np.argmax(agent.model.predict(np.expand_dims(state, axis=0), verbose=0)[0])
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

            if render:
                env.render()
                time.sleep(0.01)

            step += 1

        total_scores.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f} | Steps = {step}")

    print(f"\n Avg Reward over {episodes} episodes: {np.mean(total_scores):.2f}")
    return total_scores
os.environ["SDL_VIDEODRIVER"] = "dummy"  
# Initialize Environment and Agent
env = GameEnv()
state_size = 5
action_size = 4
agent = D3QNAgent(state_size, action_size)

# 1. LOAD WEIGHTS
# Ensure the file name matches exactly what you have on disk
try:
    print("Loading pretrained weights...")
    agent.model.load_weights("best_d3qn_16checkpoints.weights.h5")
    agent.target_model.set_weights(agent.model.get_weights())
    print("Weights loaded successfully!")
except:
    print("Weight file not found! Starting from scratch.")


agent.epsilon = 0.03
agent.epsilon_decay = 0.99
agent.epsilon_min = 0.01
# Lower learning rate slightly to prevent destroying the pretrained knowledge
agent.model.compile(optimizer=optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss='mse')

# 3. START TRAINING
print("Starting training with Physics Update...")
train_agent(env, agent, episodes=250)
env = GameEnv(render_mode=False)  # headless mode
agent = D3QNAgent(state_size=env.state_size, action_size=env.action_size)

test_scores = test_agent(env, agent, episodes=5, render=False)
os.environ["SDL_VIDEODRIVER"] = "dummy"  

env = GameEnv()  

# get proper input/output sizes
state = env.reset()
state_size = len(state)
action_size = 4  # left, right, brake

# initialize agent
agent = D3QNAgent(state_size=state_size, action_size=action_size)
agent.model.load_weights('best_d3qn_19checkpoints.weights.h5')
agent.target_model.set_weights(agent.model.get_weights())

# Use very low epsilon to focus on learned policy
agent.epsilon = 0.1 
agent.epsilon_min = 0.01
agent.epsilon_decay = 0.9999 

# train
rewards, avg, losses, max_speed, checkpoints = train_agent(env, agent, episodes=500)
