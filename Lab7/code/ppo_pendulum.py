#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import os
os.environ["XDG_RUNTIME_DIR"] = "~/.cache/xdgr"

import random
import gymnasium as gym
import numpy as np
import argparse
import wandb
import torch
from ppo import *
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")   
    parser.add_argument("--wandb-run-name", type=str, default=f"pendulum-ppo-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--entropy-weight", type=int, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--log-std-min", type=float, default=-20)
    parser.add_argument("--log-std-max", type=float, default=0)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epsilon", type=int, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=1024)  
    parser.add_argument("--update-epoch", type=float, default=64)
    parser.add_argument("--test-interval", type=int, default=5)
    parser.add_argument("--score-baseline", type=int, default=-150)
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
    while os.path.exists(f"ppo-pendulum{i}"):
        i += 1
    args.test_folder = f"ppo-pendulum{i}/videos"
    args.checkpoint_dir = f"ppo-pendulum{i}/checkpoints"

    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    get_test_env = lambda env: gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-PPO-Pendulum", name=args.wandb_run_name, save_code=True, config=vars(args))
    
    agent = PPOAgent(env, get_test_env, args)
    agent.train()