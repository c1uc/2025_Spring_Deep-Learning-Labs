#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import os
os.environ["XDG_RUNTIME_DIR"] = "~/.cache/xdgr"

from datetime import datetime
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
from typing import Tuple, Callable, List
from a2c import *

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        gae_lambda (float): GAE lambda parameter
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transitions (list): storage for the recent transitions
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, get_test_env: Callable[[], gym.Env], args=None):
        """Initialize."""
        self.env = env
        self.get_test_env = get_test_env
        self.test_env = None
        self.test_folder = args.test_folder
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.test_interval = args.test_interval
        self.wandb_run_name = args.wandb_run_name
        self.gae_lambda = args.gae_lambda
        self.use_reward_as_target = args.use_reward_as_target
        
        # device: cpu / gpu
        self.device = torch.device(args.device)
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = Actor(obs_dim, action_dim, args.activation, args.hidden_dim, args.log_std_min, args.log_std_max).to(self.device)
        self.critic = Critic(obs_dim, args.activation, args.hidden_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=self.num_episodes)
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, T_max=self.num_episodes)

        self.clip_grad = args.clip_grad
        self.clip_grad_value = args.clip_grad_value

        # transitions storage
        self.transitions: List = []

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False
        self.checkpoint_dir = args.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    @torch.no_grad()
    def get_value(self, state: np.ndarray) -> float:
        """Get the value of the state."""
        state = torch.FloatTensor(state).to(self.device)
        state = state.reshape(-1, self.obs_dim)
        return self.critic(state).item()
    
    @torch.no_grad()
    def get_dist(self, state: np.ndarray) -> torch.distributions.Normal:
        """Get the distribution of the state."""
        state = torch.FloatTensor(state).to(self.device)
        state = state.reshape(-1, self.obs_dim)
        dist = self.actor(state)
        return dist

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        dist = self.get_dist(state)
        selected_action = dist.mean if self.is_test else dist.sample()
        selected_action = selected_action.cpu().numpy()
        
        if not self.is_test:
            self.transitions.extend([state, selected_action])

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        if not self.is_test:
            next_state, reward, terminated, truncated, _ = self.env.step(action[0])
        else:
            next_state, reward, terminated, truncated, _ = self.test_env.step(action[0])
        done = terminated or truncated

        if not self.is_test:
            self.transitions.extend([reward, done, next_state])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        if not self.transitions:
            return torch.tensor(0.0), torch.tensor(0.0)

        s, a, r, d, n_s = self.transitions

        next_value = self.get_value(n_s)
        
        # Extract data from transitions
        states = torch.FloatTensor(s).to(self.device).reshape(-1, self.obs_dim)
        actions = torch.FloatTensor(a).to(self.device).reshape(-1, self.action_dim)
        
        values = self.critic(states)
        if self.use_reward_as_target:
            value_target = torch.tensor(r).float().to(self.device).reshape(-1, 1)
        else:
            value_target = torch.tensor(r + self.gamma * next_value * (1 - d)).float().to(self.device).reshape(-1, 1)
        # Compute value loss
        value_loss = F.mse_loss(values, value_target)

        # Update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_value)
        self.critic_optimizer.step()

        # Compute policy loss
        dist = self.actor(states)
        log_probs = dist.log_prob(actions)
        
        advantage = value_target - values
        
        policy_loss = -(log_probs * advantage.detach()).mean() - self.entropy_weight * dist.entropy().mean()

        # Update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_value)
        self.actor_optimizer.step()

        # Clear transitions
        self.transitions = []
        
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        return policy_loss.item(), value_loss.item()

    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        
        for ep in (pbar := tqdm(range(1, self.num_episodes))): 
            actor_losses, critic_losses, scores = [], [], []
            state, _ = self.env.reset()
            score = 0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                wandb.log({
                    "update/step": step_count,
                    "update/actor_loss": actor_loss,
                    "update/critic_loss": critic_loss,
                }) 
            
                state = next_state
                score += reward
                step_count += 1
                # W&B logging
                # if episode ends
                if done:
                    scores.append(score)
                    # print(f"Episode {ep}: Total Reward = {score}")
                    # W&B logging
                    wandb.log({
                        "train/step": step_count,
                        "train/episode": ep,
                        "train/return": score
                        })
                    
                    pbar.set_postfix(step=step_count, episode=ep, score=score)
                    
            if ep % self.test_interval == 0:
                avg_score = self.test(current_episode=ep)
                wandb.log({
                    "test/avg_score": avg_score,
                    "test/step": step_count
                })
                print(f"step: {step_count}, avg score: {avg_score}")
                
                if avg_score > -150:
                    torch.save(self.actor.state_dict(), f"{self.checkpoint_dir}/a2c_pendulum_model_ep_{ep}_step_{step_count}_score_{int(avg_score)}.pth")
                
                self.is_test = False

    @torch.no_grad()
    def test(self, epochs: int = 10, current_episode: int = 0):
        """Test the agent."""
        self.is_test = True
        self.test_env = self.get_test_env()
        self.test_env = gym.wrappers.RecordVideo(self.test_env, video_folder=self.test_folder, name_prefix=f"epoch_{current_episode}", episode_trigger=lambda x: x % 10 == 9)

        scores = []
        for _ in range(epochs):
            state, _ = self.test_env.reset()
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            scores.append(score)

        self.test_env.close()
        return np.mean(scores)


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-run-name", type=str, default=f"pendulum-a2c-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--log-std-min", type=float, default=-20)
    parser.add_argument("--log-std-max", type=float, default=0)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-episodes", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--entropy-weight", type=int, default=0)
    parser.add_argument("--test-interval", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--activation", type=str, default="mish")
    parser.add_argument("--clip-grad", type=bool, default=False)
    parser.add_argument("--clip-grad-value", type=float, default=0.5)
    parser.add_argument("--use-reward-as-target", type=bool, default=False)
    args = parser.parse_args()
    
    i = 1
    while os.path.exists(f"a2c-pendulum{i}"):
        i += 1
    args.checkpoint_dir = f"a2c-pendulum{i}/checkpoints"
    args.test_folder = f"a2c-pendulum{i}/videos"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.test_folder, exist_ok=True)
    
    if args.activation == "tanh":
        args.activation = nn.Tanh
    elif args.activation == "relu":
        args.activation = nn.ReLU
    elif args.activation == "mish":
        args.activation = nn.Mish
    else:
        raise ValueError(f"Activation function {args.activation} not supported")
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    get_test_env = lambda: gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True, config=vars(args))
    
    agent = A2CAgent(env, get_test_env, args)
    agent.train()