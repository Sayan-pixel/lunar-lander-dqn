import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import gymnasium as gym
import random
import argparse
# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Hyperparameters
MEMORY_SIZE = 100_000 #Replay Buffer Size
GAMMA = 0.995 #Discount factor
ALPHA = 5e-4 #Learning rate
BATCH_SIZE = 256 #Minibatch Size
NUM_STEPS_FOR_UPDATE = 4 #Frequency of Network updates
EPSILON_START = 1.0 #Initial exploration Probability
EPSILON_MIN = 0.05 #Minimum Exploration Probability
EPSILON_DECAY = 0.999 #Decay rate for Epsilon
num_p_av = 100 #window size for calculating average reward

# Define experience tuple for replay buffer
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

# Define Q-Network (Neural Network for Q-value approximation)
class QNetwork(nn.Module):
    def __init__(self, state_size, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
    
# Function to save model weights as a NumPy structured array
def save_model_as_array(model, save_path):
    state_dict = model.state_dict()
    numpy_representation = {key: value.cpu().numpy() for key, value in state_dict.items()}
    
    # Create a structured array for storing model weights
    dtype = [(key, np.float32, val.shape) for key, val in numpy_representation.items()]
    policy_struct = np.empty((), dtype=dtype) 
    for key, val in numpy_representation.items():
        policy_struct[key] = val
    
    np.save(save_path, policy_struct)
    print(f"✅ Model weights saved to {save_path}")

# Compute loss function for Q-learning
def compute_loss(experiences, gammaa, q_network, target_q_network):
    states, actions, rewards, next_states, dones = experiences
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
   
    with torch.no_grad():
        max_next_q_values = target_q_network(next_states).max(1)[0]
        y_targets = rewards + gammaa * max_next_q_values * (1 - dones)
    
    return nn.functional.mse_loss(q_values, y_targets)

# Update target network with soft update (polyak averaging)
def update_target_network(q_network, target_q_network):
    for target_param, q_param in zip(target_q_network.parameters(), q_network.parameters()):
        target_param.data.copy_(0.05 * q_param.data + 0.95 * target_param.data)

# Training function for the DQN agent
def train_agent(env, num_episodes=10000, max_steps=1000):
    start_time = time.time()
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
   
    # Initialize Q-Networks
    q_network = QNetwork(state_size, num_actions).to(device)
    target_q_network = QNetwork(state_size, num_actions).to(device)
    target_q_network.load_state_dict(q_network.state_dict())
   
    optimizer = optim.Adam(q_network.parameters(), lr=ALPHA)
    memory_buffer = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START
    total_point_history = []
    best_avg=0
   
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_points = 0
       
        for t in range(max_steps):
            # Epsilon-greedy policy for action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(q_network(state)).item()
           
            # Execute action and observe reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            done = terminated or truncated
            memory_buffer.append(Experience(state, action, reward, next_state, done))
            state = next_state
            total_points += reward
           
            # Perform Training Updates
            if len(memory_buffer) >= BATCH_SIZE and t % NUM_STEPS_FOR_UPDATE == 0:
                experiences = random.sample(memory_buffer, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*experiences)
               
                states = torch.stack(states).squeeze(1).to(device)
                actions = torch.tensor(actions, dtype=torch.int64, device=device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.stack(next_states).squeeze(1).to(device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)
               
                loss = compute_loss((states, actions, rewards, next_states, dones), GAMMA, q_network, target_q_network)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                update_target_network(q_network, target_q_network)
           
            if done:
                break
        #Decay Epsilon Value
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        total_point_history.append(total_points)
        avg_reward = np.mean(total_point_history[-num_p_av:])
       
        print(f"Episode {episode + 1} | Avg Reward (last {num_p_av} eps): {avg_reward:.2f}", end="\r")
       
        # Save Best Model
        if avg_reward > best_avg:
            best_avg=avg_reward
            print(f"\n\nEnvironment solved in {episode + 1} episodes! Saving model...")
            save_model_as_array(q_network,"best_policy_2205.npy")

def play_agent(env, model_path="best_policy_2205.npy", num_episodes=100):
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize the Q-network
    q_network = QNetwork(state_size, num_actions).to(device)

    # Load the structured array from the .npy file
    structured_array = np.load(model_path, allow_pickle=True)

    # Convert the structured array back into a dictionary of tensors
    model_weights = {key: torch.tensor(structured_array[key], dtype=torch.float32).to(device) 
                    for key in structured_array.dtype.names}

    # Load the weights into the Q-network
    q_network.load_state_dict(model_weights)
    q_network.eval()  # Set the network to evaluation mode

    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0

        while True:
            with torch.no_grad():
                action = torch.argmax(q_network(state)).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            total_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Score = {total_reward}")

    avg_score = np.mean(total_rewards)
    print(f"✅ Average Score over {num_episodes} episodes = {avg_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--play", action="store_true", help="Play using the trained policy")
    args = parser.parse_args()

    env = gym.make("LunarLander-v3", render_mode="rgb_array")  # Use "human" for rendering

    if args.train:
        train_agent(env)
    elif args.play:
        play_agent(env)
    else:
        print("❌ Please specify --train or --play")
    env.close()