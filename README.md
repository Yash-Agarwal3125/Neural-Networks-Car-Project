# 🚗 AI Self-Driving Car - A Deep Q-Network Adventure

Welcome to the AI Self-Driving Car project! This repository contains a complete simulation environment built with **Pygame** where a Deep Q-Network (DQN) agent, powered by **TensorFlow/Keras**, learns to navigate a challenging racetrack. The goal is to train an AI that can complete laps autonomously by learning from its own successes and failures.

This project utilizes an advanced **Pre-training and Fine-tuning** methodology, which combines the best of imitation learning and reinforcement learning to create a highly skilled and robust driving agent.

---

## 📋 Table of Contents

- [Project Status](#-project-status)
- [Timeline and Development Journey](#-timeline-and-development-journey)
- [About The Project](#-about-the-project)
- [File Structure](#-file-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [Phase 1: Manual Play & Data Collection](#phase-1-manual-play--data-collection)
  - [Phase 2: Training the Agent](#phase-2-training-the-agent)
  - [Phase 3: Running the Trained Agent](#phase-3-running-the-trained-agent)
- [Training Methodology](#-training-methodology)
- [Contributing](#-contributing)

---

##  STATUS Project Status

**Current State: Proof of Concept**

This project has successfully demonstrated the core concepts of reinforcement learning and imitation learning for autonomous navigation. The agent has undergone significant training and iteration.

* **Current Capability:** The agent can consistently and autonomously navigate the initial, more straightforward sections of the track, successfully clearing the **first three checkpoints**.
* **Current Limitation:** The agent has not yet learned to complete a full lap. It currently struggles with the more complex series of turns in the middle of the track.
* **Next Steps:** Further training and hyperparameter tuning, particularly with the reward function and network architecture, are required for the agent to master the entire circuit.

---

## 🕒 Timeline and Development Journey

This project was developed over the course of approximately **two weeks**, involving multiple stages of debugging, refactoring, and iterative training.

* **Initial Setup and Environment (Approx. 2-3 days):** The foundational work involved setting up the Pygame simulation environment, creating the `Car` class, and implementing the `ray-casting` sensor logic.
* **Initial DQN Training (Approx. 5-6 days):** This was the most intensive phase, involving numerous training runs (many lasting **4-6 hours** each on Google Colab). A significant amount of time was dedicated to debugging the agent's behavior, such as a tendency to stand still or drive in circles, and iteratively refining the reward function to encourage productive exploration.
* **Modularization and Advanced Training (Approx. 3-4 days):** The project was restructured into a modular format with a core `game_core.py` file and separate notebooks. The final training approach, combining imitation learning (pre-training) and reinforcement learning (fine-tuning), was implemented during this phase.

**Total Estimated Training Time:** Over **40 hours** of cumulative model training time, primarily conducted on Google Colab to leverage free GPU resources.

---

## 🤖 About The Project

This project explores the fascinating world of reinforcement learning by tackling a classic challenge: teaching an AI to drive. The agent perceives its environment through three forward-facing "ray-casting" sensors and its current speed. Based on this input, it must decide whether to turn left, turn right, or brake.

The agent is trained using a **Deep Q-Network (DQN)**, a type of neural network that learns to predict the value of taking a certain action in a given state. By rewarding progress and penalizing crashes, the agent gradually builds an optimal driving policy.

## 📁 File Structure

Your project is organized into modular components for clarity and ease of use:

* **`Core_Game_Parts.py`**: The heart of the simulation. This Python module contains all the shared logic, including the global constants, the `Car` class, and the `ray_casting` sensor function.
* **`User_Mode.ipynb`**: A notebook for you to drive the car manually.
* **`User_Training_Weights.ipynb`**: The notebook for the advanced **Pre-training + Fine-tuning** workflow. This is where you'll collect expert data and fine-tune the agent.
* **`Train_Agent.ipynb`**: Contains the primary reinforcement learning loop for training the agent from scratch (the more "tedious" but fundamental method).
* **`Run_Model.ipynb`**: The final presentation notebook. Use this to load a trained weights file and watch your AI drive the track.
* **`Track_images/`**: A folder containing the assets for the simulation, such as `car.png` and `track.png`.
* **`expert_data.npy` / `pretrain_data.npy`**: Data files generated when you drive the car manually, used for imitation learning.
* **`*.weights.h5`**: The saved weights files for your trained neural network models. These are the "brains" of your AI.
* **`requirements.txt`**: A file listing all the necessary Python packages to run the project.

## 🚀 Getting Started

Follow these steps to get the simulation running on your local machine.

### Prerequisites

* Python 3.10 or newer
* `pip` and `venv` (usually included with Python)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/your_repository_name.git)
    cd your_repository_name
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv .venv

    # Activate it (Windows)
    .\.venv\Scripts\activate

    # Activate it (macOS/Linux)
    source .venv/bin/activate
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 Usage

The project is broken down into distinct phases, each with its own notebook.

### Phase 1: Manual Play & Data Collection

To drive the car yourself or to collect data for imitation learning, run the **`User_Mode.ipynb`** or **`User_Training_Weights.ipynb`** notebook. This will open a Pygame window where you can control the car with the arrow keys.

### Phase 2: Training the Agent

To train your own agent, open and run the cells in the **`Train_Agent.ipynb`** notebook (for standard RL) or follow the advanced workflow in **`User_Training_Weights.ipynb`**.

### Phase 3: Running the Trained Agent

This is the most exciting part! To see your best-trained agent in action, use the **`Run_Model.ipynb`** notebook.

1.  **Open `Run_Model.ipynb`**.
2.  Find the cell at the bottom that specifies the weights file.
3.  **Update the filename** to your best-performing model, for example:
    ```python
    # IMPORTANT: Change this to the name of your saved weights file!
    WEIGHTS_FILENAME = "finetuned_weights_episode_550.weights.h5"
    ```
4.  Run the cell. A Pygame window will open, and you can watch your fully trained AI navigate the track on its own!

## 🧠 Training Methodology

This project uses a powerful two-stage training process:

1.  **Phase 1: Pre-training (Behavioral Cloning)**
    The model is first trained on data collected from a human expert (you!). This gives the agent a solid baseline understanding of how to drive, drastically reducing the initial random exploration phase.

2.  **Phase 2: Fine-tuning (Deep Q-Learning)**
    The pre-trained agent is then placed in the environment to learn on its own. It uses the DQN algorithm to explore the track, learn from its mistakes, and refine its driving policy to a level that can surpass the human expert.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
