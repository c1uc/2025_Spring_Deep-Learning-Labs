import torch
import torch.nn as nn
from typing import List
import numpy as np

def init(layer, init_w: float = 3e-3, out_dim: int = 128):
    """Initialize the weights and bias in [-init_w, init_w]."""
    if isinstance(layer, nn.Linear):
        layer.weight.data.uniform_(-init_w, init_w)
        layer.bias.data.uniform_(-init_w, init_w)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = None

    def apply_init(self, init_w: float = 3e-3, out_dim: int = 128):
        self.apply(lambda layer: init(layer, init_w, out_dim))


class Actor(BaseModel):
    def __init__(self, in_dim: int, out_dim: int, log_std_min: int = -20, log_std_max: int = 0, action_scale: float = 1.0):
        """Initialize."""
        super().__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        
        self.mean_fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Tanh(),
        )
        
        self.log_std_fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

        self.apply_init(out_dim=out_dim)

        self.action_scale = action_scale
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        #############################
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        
        x = self.mean_fc(state)
        dist = torch.distributions.Normal(x * torch.tensor(np.array(self.action_scale)).to(state.device), torch.exp(torch.clamp(self.log_std_fc(state), self.log_std_min, self.log_std_max)))
        action = dist.sample()

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
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.apply_init(out_dim=1)
        
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############

        value = self.nn(state)

        #############################

        return value
