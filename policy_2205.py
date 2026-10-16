import numpy as np
import torch
import torch.nn as nn

# Define the QNetwork class matching the training architecture
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

# Cache model instance to avoid expensive re-instantiation and state_dict reloading on every timestep
_cached_network = None
_cached_policy_id = None

# Define the policy_action function required by evaluate_agent.py
def policy_action(policy, observation, device="cpu"):
    global _cached_network, _cached_policy_id

    # Initialize and load weights once per evaluation session (or whenever policy changes)
    if _cached_network is None or _cached_policy_id != id(policy):
        state_size = len(observation)
        num_actions = 4  # LunarLander-v3 has 4 discrete actions
        _cached_network = QNetwork(state_size, num_actions).to(device)
        
        # Reconstruct the state_dict from the structured array
        state_dict = {
            key: torch.tensor(policy[key], dtype=torch.float32).to(device)
            for key in policy.dtype.names
        }
        _cached_network.load_state_dict(state_dict)
        _cached_network.eval()
        _cached_policy_id = id(policy)

    # Convert observation to a tensor and predict action
    with torch.no_grad():
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = _cached_network(observation_tensor)
        action = torch.argmax(q_values).item()

    return action