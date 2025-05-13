#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh


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
from a2c import Actor, Critic
import os

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
        
        # device: cpu / gpu
        self.device = torch.device(args.device)
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim, action_scale=2.0, log_std_min=-20, log_std_max=0).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # transitions storage
        self.transitions: List[dict] = []

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False
        self.checkpoint_dir = args.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            self.transitions.append({
                'state': state,
                'action': selected_action,
            })

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        if not self.is_test:
            next_state, reward, terminated, truncated, _ = self.env.step(action)
        else:
            next_state, reward, terminated, truncated, _ = self.test_env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transitions[-1].update({
                'next_state': next_state,
                'reward': reward,
                'done': done
            })

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        if not self.transitions:
            return torch.tensor(0.0), torch.tensor(0.0)

        # Extract data from transitions
        states = torch.stack([t['state'] for t in self.transitions])
        actions = torch.stack([t['action'] for t in self.transitions])
        rewards = torch.tensor([t['reward'] for t in self.transitions], device=self.device).float().reshape(-1, 1)
        dones = torch.tensor([t['done'] for t in self.transitions], device=self.device).reshape(-1, 1)
        
        # Compute next values
        next_states = torch.FloatTensor(np.array([t['next_state'] for t in self.transitions])).to(self.device)
        masks = ~dones
        
        with torch.no_grad():
            next_values = self.critic(next_states)
        
        values = self.critic(states)
        
        # Compute GAE
        returns = rewards + self.gamma * next_values * masks
        
        # Compute value loss
        value_loss = F.mse_loss(values, returns.reshape(-1, 1))

        # Update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Compute policy loss
        _, dist = self.actor(states)
        log_probs = dist.log_prob(actions)
        advantages = returns - values.detach()
        
        policy_loss = -(log_probs * advantages.detach()).mean() - self.entropy_weight * dist.entropy().mean()

        # Update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Clear transitions
        self.transitions = []

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
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--entropy-weight", type=int, default=1e-2)
    parser.add_argument("--test-interval", type=int, default=10)
    parser.add_argument("--test-folder", type=str, default="a2c-pendulum/videos")
    parser.add_argument("--checkpoint-dir", type=str, default="a2c-pendulum/checkpoints")
    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    get_test_env = lambda: gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
    
    agent = A2CAgent(env, get_test_env, args)
    agent.train()