import os
import time
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import multiprocessing as mp
import pygame
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.ticker import MaxNLocator
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
BG_COLOR = (120, 120, 120)  # The GREY color of the WALL/BACKGROUND
DRAW_COLOR = (50, 50, 50)   # The DARK GREY color of the ROAD
CAR_WIDTH, CAR_HEIGHT = 20, 40
DEFAULT_START_X, DEFAULT_START_Y = 1050, 550 # A known SAFE starting position
DEFAULT_START_ANGLE = 180 # Pointing left towards the first turn
ACCELERATION = 0.05
BRAKE_FORCE = 0.1
MAX_SPEED = 5.00
FRICTION = 0.025
MIN_TURN_ANGLE = 1.5
MAX_TURN_ANGLE = 2.0
CAR_IMAGE_PATH = "Track_images/car.png"
TRACK_IMAGE_PATH = "Track_images/track1.png"

checkpoint_data = [
    (834, 520, 10, 120, 0), (600, 540, 10, 120, 0), (110, 569, 10, 120, 90),
    (285, 483, 10, 120, 0), (366, 314, 10, 120, 0), (355, 173, 10, 120, 0),
    (450, 109, 10, 120, 0), (606, 170, 10, 120, 0), (818, 91, 10, 120, 0),
    (1127, 88, 10, 120, 310), (1094, 270, 10, 120, 0), (920, 346, 10, 120, 45)
]

# --- CORE CLASSES AND FUNCTIONS (Corrected) ---
class Car:
    def __init__(self, image_path, x, y, angle=0, speed=0):
        if pygame.display.get_init():
             self.original_image = pygame.transform.scale(pygame.image.load(image_path).convert_alpha(), (CAR_WIDTH, CAR_HEIGHT))
        else:
             self.original_image = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
        self.x, self.y, self.angle, self.speed = x, y, angle, speed
        self.rect = self.original_image.get_rect(center=(self.x, self.y))
    def move(self):
        rad = np.radians(self.angle)
        self.x += self.speed * np.cos(rad); self.y -= self.speed * np.sin(rad)
        self.rect.center = (self.x, self.y)

def ray_casting(car, track_surface):
    distances = []
    for angle in [-45, 0, 45]:
        ray_angle, (ray_x, ray_y), dist = car.angle + angle, (car.x, car.y), 0
        while dist < 200:
            rad = np.radians(ray_angle)
            ray_x += np.cos(rad); ray_y -= np.sin(rad)
            dist += 1
            if not (0 <= ray_x < SCREEN_WIDTH and 0 <= ray_y < SCREEN_HEIGHT): break
            try:
                # BUG FIX: Stop at the WALL (BG_COLOR)
                if track_surface.get_at((int(ray_x), int(ray_y)))[:3] == DRAW_COLOR: break
            except (IndexError, pygame.error): break
        distances.append(dist)
    return distances, []

def model_game_step(action, car, track_surface, current_checkpoint):
    done, reward = False, car.speed * 0.1
    current_state, _ = ray_casting(car, track_surface)
    reward += current_state[1] * 0.01
    car.speed += ACCELERATION
    if car.speed > 0:
        turn = MAX_TURN_ANGLE - (car.speed/MAX_SPEED)*(MAX_TURN_ANGLE-MIN_TURN_ANGLE)
        if action == 0: car.angle += turn
        elif action == 1: car.angle -= turn
    if action == 2: car.speed -= BRAKE_FORCE
    car.speed -= FRICTION; car.speed = max(0, min(car.speed, MAX_SPEED)); car.move()
    checkpoint_rects = [pygame.Rect(x,y,w,h) for x,y,w,h,a in checkpoint_data]
    if current_checkpoint < len(checkpoint_rects):
        if car.rect.colliderect(checkpoint_rects[current_checkpoint]):
            current_checkpoint += 1; reward += 1000
    try:
        # BUG FIX: Crash on the WALL (BG_COLOR)
        if track_surface.get_at((int(car.x), int(car.y)))[:3] == DRAW_COLOR: done = True
    except (IndexError, pygame.error): done = True
    if done: reward = -100
    new_state, _ = ray_casting(car, track_surface)
    return new_state, done, reward, current_checkpoint

# ==============================================================================
# CELL 3: ACTOR-CRITIC LOGIC & PLOTTING
# ==============================================================================
def model_fn():
    net = Sequential([
        Dense(64, activation='relu', input_shape=(4,)),
        Dense(64, activation='relu'),
        Dense(3, activation='linear')
    ])
    return net

def run_actor(actor_id, experience_queue, weights_path, stop_event):
    # --- FIX: Isolate this process to the CPU to prevent GPU deadlocks ---
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    print(f"[Actor {actor_id}] Started.")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()

    track_surface = pygame.image.load(TRACK_IMAGE_PATH).convert()
    actor_model = model_fn()
    epsilon = 0.1 + (random.random() * 0.9)
    epsilon_min, epsilon_decay = 0.01, 0.9998
    episode_count = 0

    while not stop_event.is_set():
        if episode_count % 10 == 0:
            try:
                actor_model.load_weights(weights_path)
            except Exception:
                pass

        car = Car(CAR_IMAGE_PATH, DEFAULT_START_X, DEFAULT_START_Y, DEFAULT_START_ANGLE)
        ckpt, total_r, max_s = 0, 0, 0
        dist, _ = ray_casting(car, track_surface)
        state = np.array(dist + [car.speed / MAX_SPEED])

        for step in range(5000):
            print(f"[Actor {actor_id}] Episode {episode_count}, Step {step}: Predicting action...")

            action = random.randrange(3) if np.random.rand() <= epsilon else np.argmax(actor_model.predict(np.reshape(state, [1, 4]), verbose=0)[0])
            dist_next, done, reward, new_ckpt = model_game_step(action, car, track_surface, ckpt)
            next_state = np.array(dist_next + [car.speed / MAX_SPEED])

            experience_queue.put(('experience', (state, action, reward, next_state, done)))

            state, ckpt = next_state, new_ckpt
            total_r += reward
            max_s = max(max_s, car.speed)

            if done:
                break

        experience_queue.put(('summary', (total_r, ckpt, max_s)))

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        episode_count += 1

    print(f"[Actor {actor_id}] Stopping.")

def run_learner(experience_queue, weights_path, stop_event, total_episodes):
    print("[Learner] Started. Initializing...")
    learner_model = model_fn()
    learner_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    target_model = model_fn()
    target_model.set_weights(learner_model.get_weights())
    learner_model.save_weights(weights_path)
    
    memory = deque(maxlen=50000)
    batch_size = 256
    episodes_completed, train_steps = 0, 0
    history = {'scores': [], 'checkpoints': [], 'max_speed': [], 'loss': []}

    print("[Learner] Ready. Waiting for data...")
    while episodes_completed < total_episodes:
        while not experience_queue.empty():
            msg_type, data = experience_queue.get()
            if msg_type == 'experience': memory.append(data)
            elif msg_type == 'summary':
                score, ckpt, max_speed = data
                history['scores'].append(score); history['checkpoints'].append(ckpt); history['max_speed'].append(max_speed)
                episodes_completed += 1
        
        if len(memory) < batch_size * 4:
            time.sleep(0.1); continue
            
        minibatch = random.sample(memory, batch_size)
        
        states_list, actions_list, rewards_list, next_states_list, dones_list = [], [], [], [], []
        for state, action, reward, next_state, done in minibatch:
            states_list.append(state); actions_list.append(action); rewards_list.append(reward)
            next_states_list.append(next_state); dones_list.append(done)

        states = np.array(states_list); actions = np.array(actions_list); rewards = np.array(rewards_list)
        next_states = np.array(next_states_list); dones = np.array(dones_list)
        
        next_actions = np.argmax(learner_model.predict(next_states, verbose=0, batch_size=batch_size), axis=1)
        next_q = target_model.predict(next_states, verbose=0, batch_size=batch_size)
        target_q = np.array([next_q[i][next_actions[i]] for i in range(batch_size)])
        targets = rewards + 0.99 * target_q * (1 - dones)
        
        current_q = learner_model.predict(states, verbose=0, batch_size=batch_size)
        for i, action in enumerate(actions): current_q[i][action] = targets[i]
            
        loss_history = learner_model.fit(states, current_q, epochs=1, verbose=0, batch_size=batch_size)
        history['loss'].append(loss_history.history['loss'][0])
        train_steps += 1
        
        if train_steps % 10 == 0: target_model.set_weights(learner_model.get_weights())
        if train_steps % 20 == 0: learner_model.save_weights(weights_path)
        
        print(f"[Learner] Progress: {episodes_completed}/{total_episodes} episodes. Training steps: {train_steps}", end='\r')

    print("\n[Learner] Training complete. Stopping actors...")
    stop_event.set()
    learner_model.save_weights("final_a3c_weights.h5")
    print("Final weights saved to 'final_a3c_weights.h5'")
    return history

def plot_results(history):
    print("Generating plots...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(4, 1, figsize=(12, 24))
    (ax1, ax2, ax3, ax4) = axes
    
    ax1.plot(history['scores'], label='Score per Episode', color='royalblue')
    ax1.set(title='Agent Score Over Time', xlabel='Episode', ylabel='Total Reward')
    ax1.legend()

    ax2.plot(history['max_speed'], label='Max Speed', color='purple')
    ax2.set(title='Max Speed Achieved per Episode', xlabel='Episode', ylabel='Max Speed')
    ax2.legend()
    
    ax3.plot(history['loss'], label='Training Loss', color='orangered', alpha=0.7)
    ax3.set(title='Model Loss Over Time', xlabel='Training Step', ylabel='MSE Loss')
    ax3.legend()

    episodes_range = range(len(history['checkpoints']))
    ax4.bar(episodes_range, history['checkpoints'], color='forestgreen', label='Checkpoints')
    ax4.set(title='Checkpoints Cleared per Episode', xlabel='Episode', ylabel='Checkpoints Cleared')
    ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax4.legend()

    plt.tight_layout()
    plt.show()
# ==============================================================================
# CELL 4: MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    # Use 1 less than your number of logical cores (e.g., 8 cores -> 7 actors)
    NUM_ACTORS = mp.cpu_count() - 1 
    TOTAL_EPISODES_TO_TRAIN = 200
    WEIGHTS_FILE = "a3c_shared_weights.h5"

    # --- Setup and Run ---
    # The 'if __name__ == "__main__"' is crucial for multiprocessing to work correctly in scripts/notebooks
    with mp.Manager() as manager:
        experience_queue = manager.Queue()
        stop_event = manager.Event()
        actors = []
        for i in range(NUM_ACTORS):
            actor = mp.Process(target=run_actor, args=(i, experience_queue, WEIGHTS_FILE, stop_event))
            actors.append(actor)
            actor.start()

        training_history = run_learner(experience_queue, WEIGHTS_FILE, stop_event, TOTAL_EPISODES_TO_TRAIN)
        
        for actor in actors:
            actor.join()

    if training_history:
        plot_results(training_history)

    print("All processes finished. Exiting.")
# --- DIAGNOSTIC CELL ---
def simple_actor_test():
    print("--- Running Diagnostic Test ---")
    # Setup a headless pygame environment
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    
    # Load the track
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    track_surface = pygame.image.load(TRACK_IMAGE_PATH).convert()
    
    
    # Spawn a car at the start position
    car = Car(CAR_IMAGE_PATH, DEFAULT_START_X, DEFAULT_START_Y)
    
    # Get the color of the pixel directly under the car
    pixel_color = track_surface.get_at((int(car.x), int(car.y)))[:3]
    
    print(f"Car spawned at: ({int(car.x)}, {int(car.y)})")
    print(f"Color under the car: {pixel_color}")
    
    # Check this color against your constants
    print(f"Your Road Color (DRAW_COLOR) is: {DRAW_COLOR}")
    print(f"Your Wall Color (BG_COLOR) is: {BG_COLOR}")
    
    if pixel_color == BG_COLOR:
        print("\nDIAGNOSIS CONFIRMED: The car spawns on the road color.")
        print("Your crash logic incorrectly treats this as a crash, causing the actors to fail.")
    else:
        print("\nDIAGNOSIS FAILED: Something else is wrong.")

# Run the test
simple_actor_test()
