import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from model import Actor, Critic, compute_gae
import numpy as np
import gymnasium as gym
from typing import List, Tuple, Callable
from tqdm import tqdm
import wandb
import os

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, get_test_env: Callable[[gym.Env], gym.Env], args):
        """Initialize."""
        self.env_name = env.unwrapped.spec.id
        self.env = env
        self.get_test_env = get_test_env
        self.test_env = None
        self.test_folder = args.test_folder
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.test_interval = args.test_interval
        self.wandb_run_name = args.wandb_run_name
        # device: cpu / gpu
        self.device = torch.device(args.device)
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.obs_dim = obs_dim
        self.actor = Actor(obs_dim, action_dim, activation=args.activation, hidden_dim=args.hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, activation=args.activation, hidden_dim=args.hidden_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=self.num_episodes)
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, T_max=self.num_episodes)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False
        self.gradient_clip = args.gradient_clip
        self.gradient_clip_value = args.gradient_clip_value
        self.score_baseline = args.score_baseline
        self.checkpoint_dir = args.checkpoint_dir
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @torch.no_grad()
    def get_dist(self, state: np.ndarray) -> torch.distributions.Normal:
        state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        dist = self.actor(state)
        return dist
    
    @torch.no_grad()
    def get_value(self, state: np.ndarray) -> float:
        state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        value = self.critic(state).detach().cpu().numpy()
        return value
        
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        dist = self.get_dist(state)
        selected_action = dist.mean if self.is_test else dist.sample()

        if not self.is_test:
            value = self.get_value(state)
            self.states.append(state)
            self.actions.append(selected_action.cpu().numpy())
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action).cpu().numpy())

        return selected_action.cpu().ravel().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        if not self.is_test:
            next_state, reward, terminated, truncated, _ = self.env.step(action)
        else:
            next_state, reward, terminated, truncated, _ = self.test_env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(reward)
            self.masks.append(1 - done)

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state).detach().cpu().numpy()

        advantages = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.from_numpy(np.concatenate(self.states)).view(-1, self.obs_dim).float().to(self.device)
        actions = torch.from_numpy(np.vstack(self.actions)).to(self.device)
        advantages = torch.from_numpy(np.concatenate(advantages)).to(self.device).float()
        values = torch.from_numpy(np.concatenate(self.values)).to(self.device).float()
        log_probs = torch.from_numpy(np.vstack(self.log_probs)).to(self.device)

        returns = values + advantages

        advantages = advantages - advantages.mean()
        advantages = advantages / advantages.std()

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            ############TODO#############

            actor_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv).mean()
            
            #############################

            # critic_loss
            ############TODO#############
            
            val = self.critic(state)
            val_target = return_
            critic_loss = F.mse_loss(val, val_target)

            #############################
            
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            if self.gradient_clip:
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.gradient_clip_value)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.gradient_clip:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.gradient_clip_value)
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        self.actor_scheduler.step()
        self.critic_scheduler.step()

        return actor_loss, critic_loss

    def train(self):
        """Train the PPO agent."""
        self.is_test = False

        state, _ = self.env.reset()
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        for ep in (pbar := tqdm(range(1, self.num_episodes))):
            score = 0
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                # if episode ends
                if done[0][0]:
                    pbar.set_postfix(step=self.total_step, episode=episode_count, score=score)
                    wandb.log({
                        "train/step": self.total_step,
                        "train/episode": episode_count,
                        "train/return": score
                    })
                    
                    episode_count += 1
                    state, _ = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    score = 0
                    
                if self.total_step % 500_000 == 0:
                    fname = f"{self.total_step // 1_000_000}p5m" if self.total_step % 1_000_000 != 0 else f"{self.total_step // 1_000_000}m"
                    torch.save(self.actor.state_dict(), f"{self.checkpoint_dir}/ppo_{self.env_name}_model_step_{fname}.pth")
                    
            actor_loss, critic_loss = self.update_model(next_state)
            
            wandb.log({
                "update/step": self.total_step,
                "update/actor_loss": actor_loss,
                "update/critic_loss": critic_loss
            })
            
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
            if ep % self.test_interval == 0:
                avg_score = self.test(current_episode=ep)
                wandb.log({
                    "test/avg_score": avg_score,
                    "test/step": self.total_step
                })
                print(f"step: {self.total_step}, avg score: {avg_score}")
                
                if avg_score > self.score_baseline:
                    torch.save(self.actor.state_dict(), f"{self.checkpoint_dir}/ppo_{self.env_name}_model_ep_{ep}_step_{self.total_step}_score_{int(avg_score)}.pth")
                
                self.is_test = False

        # termination
        self.env.close()

    def test(self, epochs: int = 10, current_episode: int = 0):
        """Test the agent."""
        self.is_test = True
        self.test_env = self.get_test_env(self.env)
        self.test_env = gym.wrappers.RecordVideo(self.test_env, video_folder=self.test_folder, name_prefix=f"epoch_{current_episode}", episode_trigger=lambda x: x == 0)

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