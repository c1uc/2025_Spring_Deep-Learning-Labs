import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque
import random


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AtariPreprocessor:
    """
    Preprocesing the state input of DQN for Atari
    """

    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque(
            [frame for _ in range(self.frame_stack)], maxlen=self.frame_stack
        )
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)
    

class ReplayBuffer:
    """
    Replay buffer for storing transitions
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size), None, None

class PrioritizedReplayBuffer:
    """
    Prioritizing the samples in the replay memory by the Bellman error
    See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, error=np.inf):
        ########## YOUR CODE HERE (for Task 3) ##########

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = abs(error)
        self.pos = (self.pos + 1) % self.capacity

        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########

        probs = self.priorities[:len(self.buffer)]

        if np.any(probs):
            probs = np.where(probs == np.inf, 10, 1).astype(np.float32)
        else:
            probs = probs ** self.alpha + self.epsilon

        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.sum()

        return samples, weights, indices
        
        ########## END OF YOUR CODE (for Task 3) ##########

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########

        self.priorities[indices] = abs(errors) + self.epsilon

        ########## END OF YOUR CODE (for Task 3) ##########
        return


class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

        self.apply(init_weights)

    def forward(self, x):
        return self.nn(x)


class DQN_Atari(nn.Module):
    def __init__(self, num_actions):
        super(DQN_Atari, self).__init__()

        self.nn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

        self.apply(init_weights)

    def forward(self, x):
        x = x.float() / 255.0
        return self.nn(x)
