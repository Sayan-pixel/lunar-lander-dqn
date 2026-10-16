# 🚀 Lunar Lander Autonomous Agent (Deep Q-Network - DQN)

[![CI Evaluation](https://github.com/Sayan-pixel/lunar-lander-dqn/actions/workflows/evaluate.yml/badge.svg)](https://github.com/Sayan-pixel/lunar-lander-dqn/actions/workflows/evaluate.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Box2D-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Solved](https://img.shields.io/badge/Environment-Solved%20(Avg%20%2B281.8)-brightgreen.svg)]()

An autonomous deep reinforcement learning agent trained to solve the **LunarLander-v3** continuous physics environment using **Deep Q-Learning (DQN)** with **Experience Replay** and **Polyak Soft Target Updates**.

The agent achieves an average evaluation reward of **+280.47 ± 18.4** over 100 consecutive episodes (well exceeding the +200 environment solving threshold) with a **100% safe touchdown rate**.

---

## 🌟 Key Features

- **Deep Q-Network (DQN)**: Multi-layer perceptron approximating optimal state-action value functions over continuous 8-dimensional space.
- **Experience Replay Buffer**: Cyclic transition memory ($N = 100,000$) to eliminate temporal auto-correlation and guarantee i.i.d. training batches.
- **Polyak Soft Target Updates ($	au = 0.05$)**: Continuous tracking to eliminate the moving-target divergence instability common in off-policy learning.
- **Lightweight Parameter Serialization**: Serializes neural network weights into a zero-dimensional NumPy structured array (105 KB), eliminating heavy framework-specific checkpoint files.
- **Optimized Lazy Model Caching**: High-throughput inference cache that evaluates 100 complete episodes in **~31 seconds** (well within strict compute constraints).
- **Interactive Visualizer & Real-Time Telemetry**: Real-time flight dashboard with live altitude, velocities, pitch angles, and leg contact sensors.

---

## 📊 Benchmark Results

| Metric | Measured Value | Solving Benchmark | Status |
| :--- | :--- | :--- | :--- |
| **Mean Evaluation Reward** | **+281.82 ± 18.4** | ≥ +200.0 | 🌟 Solved (+81.8 margin) |
| **Safe Touchdown Rate** | **100%** | — | ✓ Zero crash landings |
| **Evaluation Speed (100 eps)** | **31.55 seconds** | ≤ 100.0s | ✓ High-performance cache |
| **Policy Weight File Size** | **105 KB** | ≤ 1 MB | ✓ Compact NumPy binary |

---

## 🛠️ Architecture & Hyperparameters

```
State (8D) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(64) → ReLU → Linear(4) → Q-Values (4D)
```

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Discount Factor ($\gamma$)** | `0.995` | Long-term horizon return weighting |
| **Learning Rate ($lpha$)** | `5e-4` | Adam optimizer learning rate |
| **Minibatch Size ($B$)** | `256` | Experience replay batch size |
| **Replay Memory Capacity ($N$)** | `100,000` | Transitions buffer size |
| **Update Frequency** | `4` steps | Interval between gradient descent steps |
| **Polyak Coefficient ($	au$)** | `0.05` | Soft target network update rate |
| **Exploration ($\epsilon$)** | `1.0 → 0.05` | Geometric decay at $0.999$ per episode |

---

## 📂 Repository Structure

```
LunarLander-DQN/
├── best_policy_2205.npy         # Serialized model parameters (105 KB)
├── policy_2205.py              # Agent policy interface with lazy model caching
├── train_agent_2205.py         # Complete training pipeline with DQN & experience replay
├── evaluate_agent.py           # Benchmark harness (evaluates 100 episodes)
├── watch_agent.py              # Interactive real-time visualizer with flight telemetry
├── play_lunar_lander.py        # Manual human control script (W, A, S, D)
├── evaluate.bat                # Windows batch shortcut for 100-episode evaluation
├── watch.bat                   # Windows batch shortcut for visual telemetry player
├── LunarLander_Methodology.pdf # Comprehensive 2-page academic methodology report
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/Sayan-pixel/lunar-lander-dqn.git
cd lunar-lander-dqn
pip install -r requirements.txt
```

### 2. Watch the Trained Agent Fly (Live Visualizer)

Run the visualizer with live flight telemetry:
```bash
python watch_agent.py
```
*(On Windows, you can simply double-click `watch.bat`)*

### 3. Run Benchmark Evaluation

Evaluate the agent over 100 episodes:
```bash
python evaluate_agent.py --filename best_policy_2205.npy --policy_module policy_2205
```
*(On Windows, you can simply double-click `evaluate.bat`)*

### 4. Train from Scratch

Train a new DQN agent using your environment:
```bash
python train_agent_2205.py --train
```

---

## 📄 Academic Methodology Report

For a detailed derivation of the Markov Decision Process (MDP), Bellman Optimality Equation, Mean Squared Bellman Error (MSBE) loss function, and theoretical convergence properties, refer to [`LunarLander_Methodology.pdf`](LunarLander_Methodology.pdf).

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
