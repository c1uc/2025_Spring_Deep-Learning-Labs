import torch
import torch.nn as nn
from typing import List
import numpy as np

def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############

    values = values + [next_value]
    gae = 0
    gae_returns = []
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns.insert(0, gae)
    
    #############################
    return gae_returns


def initialize_uniformly(layer, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight)
        nn.init.zeros_(layer.bias)

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = None

    def apply_init(self, init_w: float = 3e-3):
        self.apply(lambda layer: initialize_uniformly(layer, init_w))


class Actor(BaseModel):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module, hidden_dim: int, log_std_min: float, log_std_max: float):
        """Initialize."""
        super().__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.log_std = nn.Parameter(torch.zeros(1, out_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.apply_init()
        
        #############################
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        
        mean = self.fc(state)
        std = torch.exp(torch.clamp(self.log_std, self.log_std_min, self.log_std_max))
        dist = torch.distributions.Normal(mean, std)

        #############################

        return dist


class Critic(BaseModel):
    def __init__(self, in_dim: int, activation: nn.Module, hidden_dim: int):
        """Initialize."""
        super().__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights

        self.nn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 1),
        )

        self.apply_init()
        
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############

        value = self.nn(state)

        #############################

        return value
