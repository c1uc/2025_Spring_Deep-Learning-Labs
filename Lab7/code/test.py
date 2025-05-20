import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XDG_RUNTIME_DIR"] = "~/.cache/xdgr"

import torch
import gymnasium as gym
import numpy as np
from torch import nn
import pickle
import argparse

@torch.no_grad()
def test(model, env, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
        
    state, _ = env.reset(seed=seed)
    done = False
    score = 0
    
    while not done:
        state = torch.FloatTensor(state).reshape(1, -1).to("cuda:0")
        dist = model(state)
        action = dist.mean.cpu().numpy().flatten()
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        
        score += reward
        done = truncated or done
        
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    
    env_name = "Pendulum-v1" if "pendulum" in args.model_path else "Walker2d-v4"
    env = gym.make(env_name, render_mode="rgb_array")
    
    if "a2c" in args.model_path:
        from a2c import Actor
        
        model = Actor(env.observation_space.shape[0], env.action_space.shape[0], nn.Mish, 64, -20, 0).to("cuda:0")
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        
        seeds = [89, 90, 91, 92, 94, 95, 96, 97, 99, 103, 105, 107, 109, 114, 115, 121, 123, 124, 128, 129]
    elif "ppo" in args.model_path:
        from ppo import Actor
        model = Actor(env.observation_space.shape[0], env.action_space.shape[0], nn.Tanh, 64).to("cuda:0")
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        
        seeds = [89, 90, 91, 92, 94, 95, 96, 97, 99, 103, 105, 107, 109, 114, 115, 116, 121, 122, 123, 124]
        
    if "pendulum" not in args.model_path:
        from ppo_walker import FixedNormalizeObservation
        obs_rms = pickle.load(open("walker_obs_rms.pkl", "rb"))
        env = FixedNormalizeObservation(env, obs_rms.mean, obs_rms.var)
        
        seeds = [s + 88 for s in range(20)]
        
    env = gym.wrappers.RecordVideo(env, "test-videos1", episode_trigger=lambda x: True)
        
    scores = []
    for s in seeds:
        scores.append(test(model, env, s))
        
    env.close()
    print(np.mean(scores))
