import torch
import gymnasium as gym
import numpy as np
from torch import nn
import pickle
import argparse

def test(model_path, env_name, seed):
    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
        
    if "walker" in model_path:
        from ppo_walker import FixedNormalizeObservation
        obs_rms = pickle.load(open("walker_obs_rms.pkl", "rb"))
        env = FixedNormalizeObservation(env, obs_rms.mean, obs_rms.var)
        
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        action = model.select_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        score += reward
        
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ppo-walker1/checkpoints/model_1000000.pth")
    args = parser.parse_args()
    
    env_name = "Walker2d-v4" if "walker" in args.model_path else "Pendulum-v1"
    env = gym.make(env_name)
    
    if "a2c" in args.model_path:
        from a2c import Actor
        model = Actor(env.observation_space.shape[0], env.action_space.shape[0], nn.Tanh, 64, -20, 2)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
    elif "ppo" in args.model_path:
        from ppo import Actor
        model = Actor(env.observation_space.shape[0], env.action_space.shape[0], nn.Tanh, 64)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        
    scores = []
    for s in range(20):
        scores.append(test(model, env_name, s))
    print(np.mean(scores))
