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
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-run-name", type=str, default=f"walker-ppo-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--entropy-weight", type=int, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--log-std-min", type=float, default=-50)
    parser.add_argument("--log-std-max", type=float, default=5)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--epsilon", type=int, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=1000)  
    parser.add_argument("--update-epoch", type=float, default=64)
    parser.add_argument("--test-interval", type=int, default=5)
    parser.add_argument("--test-folder", type=str, default="ppo-walker/videos")
    parser.add_argument("--checkpoint-dir", type=str, default="ppo-walker/checkpoints")
    parser.add_argument("--score-baseline", type=int, default=2500)
    args = parser.parse_args()
    
    def wrap_env(env):
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.NormalizeReward(env)
        return env
 
    # environment
    env = gym.make("Walker2d-v4", render_mode="rgb_array")
    env = wrap_env(env)
    get_test_env = lambda: wrap_env(gym.make("Walker2d-v4", render_mode="rgb_array"))
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True)
    
    agent = PPOAgent(env, get_test_env, args)
    agent.train()