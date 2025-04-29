import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import os
import argparse
from gymnasium.wrappers import RecordVideo

from models import *

import ale_py

gym.register_envs(ale_py)

def evaluate(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env, render_mode="rgb_array")
    if args.save:
        env = RecordVideo(env, video_folder=args.output_dir, name_prefix="eval", episode_trigger=lambda x: True)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    if args.env == "ALE/Pong-v5":
        preprocessor = AtariPreprocessor()
        num_actions = env.action_space.n
    else:
        preprocessor = None
        num_actions = env.action_space.n

    if args.env == "ALE/Pong-v5":
        model = DQN_Atari(num_actions).to(device)
    else:
        model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if preprocessor:
            state = preprocessor.reset(obs)
        else:
            state = obs
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if preprocessor:
                state = preprocessor.step(next_obs)
            else:
                state = next_obs

        rewards.append(total_reward)
        print(f"Episode {ep} finished with reward {total_reward}")

    env.close()
    print(f"Average reward: {np.mean(rewards)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained .pt model"
    )
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--seed", type=int, default=187, help="Random seed for evaluation"
    )
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", choices=["ALE/Pong-v5", "CartPole-v1"], help="Environment to evaluate on")
    parser.add_argument("--save", type=bool, default=False, help="Save videos")
    args = parser.parse_args()
    evaluate(args)
