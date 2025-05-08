import torch
import torch.nn as nn
from typing import List
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
        gae_returns.insert(0, gae + values[step])
    
    #############################
    return gae_returns


def initialize_uniformly(layer, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    if isinstance(layer, nn.Linear):
        layer.weight.data.uniform_(-init_w, init_w)
        layer.bias.data.uniform_(-init_w, init_w)

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = None

    def apply_init(self, init_w: float = 3e-3):
        self.apply(lambda layer: initialize_uniformly(layer, init_w))


class Actor(BaseModel):
    def __init__(self, in_dim: int, out_dim: int, log_std_min: int = -20, log_std_max: int = 0):
        """Initialize."""
        super().__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.mean_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, out_dim),
            nn.Tanh(),
        )

        self.log_std_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, out_dim),
        )

        self.apply_init()
        
        #############################
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        
        x = self.fc(state)
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x)
        dist = torch.distributions.Normal(mean, torch.exp(log_std))
        action = dist.sample()
        action = torch.tanh(action)

        #############################

        return action, dist


class Critic(BaseModel):
    def __init__(self, in_dim: int):
        """Initialize."""
        super().__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights

        self.nn = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.apply_init()
        
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############

        value = self.nn(state)

        #############################

        return value
