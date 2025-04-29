# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import ale_py
import os
import wandb
import argparse
import datetime
import tqdm

from models import *

gym.register_envs(ale_py)


class DQNAgent:
    def __init__(self, args=None):
        self.env = gym.make(args.env_name, render_mode="rgb_array")
        self.test_env = gym.make(args.env_name, render_mode="rgb_array")

        if "ALE" in args.env_name:
            self.preprocessor = AtariPreprocessor()
        else:
            self.preprocessor = None

        self.num_actions = self.env.action_space.n

        self.device = torch.device(
            f"cuda:{args.gpu}"
            if torch.cuda.is_available() and args.gpu != -1
            else "cpu"
        )
        print(f"Using device: {self.device}")

        if "ALE" in args.env_name:
            self.q_net = DQN_Atari(self.num_actions).to(self.device)
            self.target_net = DQN_Atari(self.num_actions).to(self.device)
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            self.q_net = DQN(self.num_actions).to(self.device)
            self.target_net = DQN(self.num_actions).to(self.device)
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.ddqn = args.ddqn
        self.per = args.per
        self.exit_on_loss = args.exit_on_loss

        self.n_step_return = args.n_step_return
        if self.n_step_return > 1:
            self.n_step_memory = deque(maxlen=self.n_step_return)
            self.n_step_returns = 0

        if args.per:
            self.memory = PrioritizedReplayBuffer(args.memory_size)
        else:
            self.memory = ReplayBuffer(args.memory_size)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.loss_fn = torch.nn.MSELoss()

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_steps = 0
        self.train_steps = 0
        self.best_reward = -np.inf  # Initilized to 0 for CartPole and to -21 for Pong
        self.ewma_reward = -21 if "ALE" in args.env_name else 0
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = (
            torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def calc_n_step_return(self):
        discounted_return = 0
        gamma = 1
        for transition in self.n_step_memory:
            discounted_return += gamma * transition[2]
            gamma *= self.gamma
        return discounted_return

    def run(self, episodes=1000):
        for ep in (pbar := tqdm.tqdm(range(episodes))):
            pbar.set_postfix(
                update=self.train_steps, reward=self.ewma_reward, epsilon=self.epsilon
            )
            state, _ = self.env.reset()

            if self.preprocessor:
                state = self.preprocessor.reset(state)

            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if self.preprocessor:
                    next_state = self.preprocessor.step(next_state)

                if self.exit_on_loss and reward < 0:
                    done = True

                if self.n_step_return > 1:
                    if len(self.n_step_memory) == self.n_step_return:
                        ret = self.n_step_returns
                        s, a, r, s_, d = self.n_step_memory.popleft()
                        self.memory.add((s, a, ret, next_state, done))

                        self.n_step_returns -= r
                        self.n_step_returns /= self.gamma

                    self.n_step_returns += reward * (
                        self.gamma ** len(self.n_step_memory)
                    )
                    self.n_step_memory.append((state, action, reward, next_state, done))
                else:
                    self.memory.add((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_steps += 1
                step_count += 1

                if done:
                    if self.n_step_return > 1:
                        while len(self.n_step_memory) > 0:
                            ret = self.n_step_returns
                            s, a, r, s_, d = self.n_step_memory.popleft()
                            self.memory.add((s, a, ret, next_state, done))

                            self.n_step_returns -= r
                            self.n_step_returns /= self.gamma

                    self.n_step_returns = 0
                    break

            wandb.log(
                {
                    "train/episode": ep,
                    "train/total_reward": total_reward,
                    "train/env_step_count": self.env_steps,
                    "train/update_count": self.train_steps,
                    "train/epsilon": self.epsilon,
                }
            )

            self.ewma_reward = self.ewma_reward * 0.99 + total_reward * 0.01

            if self.env_steps % 200_000 == 0:
                model_path = os.path.join(
                    self.save_dir, f"model_step{self.env_steps}.pt"
                )
                torch.save(self.q_net.state_dict(), model_path)

            if ep % 500 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)

                wandb.log(
                    {
                        "eval/env_step_count": self.env_steps,
                        "eval/update_count": self.train_steps,
                        "eval/reward": eval_reward,
                    }
                )

    def evaluate(self, eval_steps=1):
        rewards = []
        for _ in range(eval_steps):
            state, _ = self.test_env.reset()

            if self.preprocessor:
                state = self.preprocessor.reset(state)

            done = False
            total_reward = 0

            while not done:
                state_tensor = (
                    torch.from_numpy(np.array(state))
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_state, reward, terminated, truncated, _ = self.test_env.step(
                    action
                )
                done = terminated or truncated
                total_reward += reward

                if self.preprocessor:
                    next_state = self.preprocessor.step(next_state)

                state = next_state

            rewards.append(total_reward)

        return np.mean(rewards)

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_steps += 1

        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer

        batch, weights, indices = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling

        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(
            self.device
        )
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates

        with torch.no_grad():
            if self.ddqn:
                next_actions = self.q_net(next_states).argmax(dim=1)
                next_q_values = (
                    self.target_net(next_states)
                    .gather(1, next_actions.unsqueeze(1))
                    .squeeze(1)
                )
                next_q_values = next_q_values * (1 - dones)
                target_q_values = (
                    rewards + (self.gamma**self.n_step_return) * next_q_values
                )
            else:
                next_q_values = self.target_net(next_states).max(dim=1)[0]
                next_q_values = next_q_values * (1 - dones)
                target_q_values = (
                    rewards + (self.gamma**self.n_step_return) * next_q_values
                )

        loss = 0

        if self.per:
            mse = (target_q_values - q_values) ** 2
            loss = torch.tensor(weights).to(self.device) * mse
            loss = loss.mean()
            priorities = torch.abs(target_q_values - q_values).detach().cpu().numpy()
            self.memory.update_priorities(indices, priorities)
        else:
            loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        wandb.log(
            {
                "update/loss": loss.item(),
                "update/q_values": q_values.mean().item(),
                "update/target_q_values": target_q_values.mean().item(),
                "update/step": self.train_steps,
            }
        )

        ########## END OF YOUR CODE ##########

        if self.train_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        # if self.train_steps % 1000 == 0:
        #    print(f"[Train #{self.train_steps}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    import yaml

    with open(args.config, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args = AttributeDict(args)

    wandb.init(
        project=f"DLP-Lab5-DQN-{args.env_name.replace('/', '-')}",
        name=f"{args.wandb_run_name}-per_{args.per}-n_step_{args.n_step_return}-ddqn_{args.ddqn}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        save_code=False,
        config=args,
    )
    agent = DQNAgent(args=args)
    agent.run(episodes=args.episodes)
