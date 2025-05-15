#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import os
os.environ["MUJOCO_GL"] = "egl"

from datetime import datetime
import random
import gymnasium as gym
import numpy as np
import torch
import argparse
import wandb
from ppo import *

class FixedNormalizeObservation(gym.Wrapper):
    """Wrapper that uses fixed normalization statistics."""
    def __init__(self, env, mean, var):
        super().__init__(env)
        self.mean = mean
        self.var = var

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (obs - self.mean) / np.sqrt(self.var), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return (obs - self.mean) / np.sqrt(self.var), info
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-run-name", type=str, default=f"walker-ppo-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=float, default=1500)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--entropy-weight", type=int, default=0) # entropy can be disabled by setting this to 0
    parser.add_argument("--log-std-min", type=float, default=-50)
    parser.add_argument("--log-std-max", type=float, default=5)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=int, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2048)
    parser.add_argument("--update-epoch", type=float, default=16)
    parser.add_argument("--test-interval", type=int, default=5)
    parser.add_argument("--score-baseline", type=int, default=2500)
    parser.add_argument("--gradient-clip", type=bool, default=False)
    parser.add_argument("--gradient-clip-value", type=float, default=0.5)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    if args.activation == "tanh":
        args.activation = nn.Tanh
    elif args.activation == "relu":
        args.activation = nn.ReLU
    elif args.activation == "mish":
        args.activation = nn.Mish
    else:
        raise ValueError(f"Activation function {args.activation} not supported")

    i = 1
    while os.path.exists(f"ppo-walker{i}"):
        i += 1
    args.test_folder = f"ppo-walker{i}/videos"
    args.checkpoint_dir = f"ppo-walker{i}/checkpoints"
    
    # environment
    env = gym.make("Walker2d-v4", render_mode="rgb_array")
    env = gym.wrappers.NormalizeObservation(env)
    
    # Create test env with fixed normalization
    get_test_env = lambda env: FixedNormalizeObservation(
        gym.make("Walker2d-v4", render_mode="rgb_array"),
        env.obs_rms.mean,
        env.obs_rms.var
    )
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True, config=vars(args))
    
    agent = PPOAgent(env, get_test_env, args)
    agent.train()