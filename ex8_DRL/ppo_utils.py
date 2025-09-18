import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt

import torch # need to install
import torch.nn as nn # need to install
import torch.optim as optim # need to install

from typing import Optional, Tuple, Deque
from collections import deque
from pathlib import Path
import seaborn as sns # need to install

sys.path.append(os.path.abspath(".."))
from utils.env import Env_rl_c
from utils.controller import BaseController





def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer





class PPOController(BaseController):
    def __init__(
        self,
        mdp: Env_rl_c,
        freq: float,
        learning_rate_p: float = 1e-3,
        learning_rate_v: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.1,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 1.5,
        n_steps: int = 256,      # size of collected data in each rollout
        batch_size: int = 32,     # size of mini-batch
        n_epochs: int = 10,       
        max_iterations: int = 1000, 
        norm_adv: bool = True,
        clip_vloss: bool = True,
        seed: int = None, 
        save_path: str = "checkpoints/ppo.pt",
        name: str = "PPO",
        type: str = "RL",
        verbose: bool = True,
    ) -> None:

        super().__init__(mdp, mdp.dynamics, freq, name, type, verbose)

        self.learning_rate_p = learning_rate_p
        self.learning_rate_v = learning_rate_v
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_iterations = max_iterations
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.save_path = save_path
        
        self.start_iteration = 0
        self.current_iteration = 0

        self.mdp = mdp

        self.obs_dim = 2
        self.act_dim = 1
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Actor
        self.policy_net = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.act_dim), std=0.01),
        )

        # Log of standard deviation as training parameter
        self.log_std = nn.Parameter(torch.zeros(self.act_dim))

        # Critic
        self.value_net = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.parameters = [
            {"params": self.policy_net.parameters(), "lr": self.learning_rate_p},  
            {"params": [self.log_std], "lr": self.learning_rate_p},
            {"params": self.value_net.parameters(), "lr": self.learning_rate_v}  
        ]

        # Optimizer
        self.optimizer = optim.AdamW(
            self.parameters, 
            lr=self.learning_rate_p, 
            eps=1e-5  
        )

        self.reward_history = []  
        self.value_loss_history = []
        self.policy_loss_history = []
    
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'log_std': self.log_std.detach().cpu(),
            'optimizer': self.optimizer.state_dict(),
            'reward_history': self.reward_history,
            'value_loss_history': self.value_loss_history,
            'policy_loss_history': self.policy_loss_history,
            'current_iteration': self.current_iteration,
        }
        torch.save(checkpoint, path)
        print(f"[✓] PPO model saved to: {path}")
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)

        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.log_std.data = checkpoint['log_std']
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.reward_history = checkpoint.get('reward_history', [])
        self.value_loss_history = checkpoint.get('value_loss_history', [])
        self.policy_loss_history = checkpoint.get('policy_loss_history', [])
        self.current_iteration = checkpoint.get('current_iteration', 0) 
        self.start_iteration = self.current_iteration
        
        if self.verbose:
            print(f"[✓] PPO model loaded from: {path} (iteration={self.current_iteration})")

    def _action_reverse_tf(self, action_scaled):

        return self.mdp.input_lbs + (action_scaled+1) * (self.mdp.input_ubs-self.mdp.input_lbs) / 2

    def _get_action(self, state):

        state_tensor = torch.tensor(state, dtype=torch.float32)

        action_mean = self.policy_net(state_tensor)
        action_std = torch.exp(self.log_std)
        
        # Clamp std
        action_std = action_std.clamp(1e-3, 0.5)

        # Action distribution
        dist = torch.distributions.Normal(action_mean, action_std)

        # Unsquashed action sampling
        action = dist.sample()

        # Squashed action
        action_squashed = torch.tanh(action)

        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        log_prob = dist.log_prob(action).sum(dim=-1)
        log_prob_squashed = log_prob - torch.sum(torch.log(1 - action_squashed.pow(2) + 1e-7), dim=-1)

        entropys = dist.entropy().sum(dim=-1)
        value = self.value_net(state_tensor)

        return action, action_squashed, log_prob_squashed, entropys, value
    
    def evaluate_actions(self, states, actions_unsquashed):
        """
        Evaluate actions in current policy
        """
        # (batch_size, obs_dim)
        # (batch_size, act_dim)
        action_mean = self.policy_net(states)
        action_std = torch.exp(self.log_std).clamp(1e-3, 0.5)
        dist = torch.distributions.Normal(action_mean, action_std)

        # do log_prob for unsquashed actions
        log_prob = dist.log_prob(actions_unsquashed).sum(dim=-1)

        # apply squashing
        action_squashed = torch.tanh(actions_unsquashed)
        log_prob_squashed = log_prob - torch.sum(torch.log(1 - action_squashed.pow(2) + 1e-7), dim=-1)

        entropys = dist.entropy().sum(dim=-1)
        value = self.value_net(states)
        return log_prob_squashed, entropys, value

    def _compute_advantages(self, rewards, values, gamma, lam):

        advantages = np.zeros_like(rewards, dtype=np.float32)

        last_adv = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            advantages[t] = last_adv = delta + gamma * lam * last_adv

        return advantages

    def _collect_trajectories(self, n_steps: int):
        states = []
        actions_unsquashed = []
        actions_squashed = []
        rewards = []
        log_probs_squashed = []
        values = []

        steps_collected = 0
        state = self.mdp.init_state #np.random.uniform(self.env.state_lbs, self.env.state_ubs)

        while steps_collected < n_steps:
            done = False
            while not done and steps_collected < n_steps:
                # 1. Get action
                action_unsquashed, action_squashed, log_prob_squashed, _, value = self._get_action(state)

                action_squashed = action_squashed.detach().numpy()
                action_unsquashed = action_unsquashed.detach().numpy()
                log_prob_squashed = log_prob_squashed.detach().numpy()
                value = value.detach().numpy()

                # 2. Take action in env
                action_env = self._action_reverse_tf(action_squashed)
                done, next_state, reward = self.mdp.one_step_forward(state, action_env)

                # 3. Store data
                states.append(state)
                actions_unsquashed.append(action_unsquashed)
                actions_squashed.append(action_squashed)
                rewards.append(reward)
                log_probs_squashed.append(log_prob_squashed)
                values.append(value)

                state = next_state
                steps_collected += 1

            # restart from a new initial state if episode ends early
            state = self.mdp.init_state #np.random.uniform(self.env.state_lbs, self.env.state_ubs)

        # 4. Bootstrap final value (for GAE)
        final_value = self.value_net(torch.tensor(state, dtype=torch.float32)).detach().numpy()
        values.append(final_value)

        return (
            np.array(states),
            np.array(actions_unsquashed),
            np.array(actions_squashed),
            np.array(rewards),
            np.array(log_probs_squashed),
            np.array(values),
        )

    def setup(self) -> None:

        for iteration in range(self.max_iterations):

            "collect data from rollout"
            states, actions_unsquashed, actions_squashed, rewards, old_log_probs_squashed, values = self._collect_trajectories(self.n_steps)

            "GAE and return"
            advantages = self._compute_advantages(
                rewards=rewards,
                values=values.squeeze(),
                gamma=self.gamma,
                lam=self.gae_lambda
            )
            returns = advantages + values[:-1]

            "transform to tensor"
            states_t = torch.tensor(states, dtype=torch.float32)
            actions_unsquashed_t = torch.tensor(actions_unsquashed, dtype=torch.float32)
            actions_squashed_t = torch.tensor(actions_squashed, dtype=torch.float32)
            old_log_probs_squashed_t = torch.tensor(old_log_probs_squashed, dtype=torch.float32)
            advantages_t = torch.tensor(advantages, dtype=torch.float32)
            returns_t = torch.tensor(returns, dtype=torch.float32)
            
            # if self.verbose:
            #     print("states_t:", states_t)
            #     print("actions_unsquashed_t:", actions_unsquashed_t)
            #     print("actions_squashed_t:", actions_squashed_t)
            #     print("old_log_probs_squashed_t:", old_log_probs_squashed_t)
            #     print("advantages_t:", advantages_t)
            #     print("values:", values)
            #     print("returns_t:", returns_t)

            "update loop"
            dataset_size = len(states)
            indices = np.arange(dataset_size)
            #clipfracs = []
            
            for epoch in range(self.n_epochs):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, self.batch_size):
                    end = start + self.batch_size
                    mb_inds = indices[start:end]

                    if len(mb_inds) == 0:
                        continue  # incase void batch

                    #if actions_scaled_t.dim() == 0:
                    #    actions_scaled_t = actions_scaled_t.unsqueeze(0)

                    # mini-batch
                    mb_states = states_t[mb_inds]
                    mb_actions_unsquashed = actions_unsquashed_t[mb_inds]
                    mb_old_log_probs_squashed = old_log_probs_squashed_t[mb_inds]
                    mb_advantages = advantages_t[mb_inds]
                    mb_returns = returns_t[mb_inds]

                    # forward
                    new_log_probs_squashed, new_entropys, new_values = self.evaluate_actions(mb_states, mb_actions_unsquashed)

                    # ratio
                    logratio = new_log_probs_squashed - mb_old_log_probs_squashed
                    ratio = torch.exp(logratio)

                    #with torch.no_grad():
                    #    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    #    old_approx_kl = (-logratio).mean()
                    #    approx_kl = ((ratio - 1) - logratio).mean()
                    #    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                    
                    if self.norm_adv:
                        # only do adv normalization when has more than 1 element
                        if len(mb_advantages) > 1:
                            mb_advantages_std = torch.clamp(mb_advantages.std(unbiased=False), min=1e-8)
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_advantages_std

                    # clip
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)

                    # policy loss
                    policy_loss_1 = - ratio * mb_advantages
                    policy_loss_2 = - clipped_ratio * mb_advantages
                    policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                    # value loss
                    if self.clip_vloss:
                        value_loss_unclipped = (new_values - mb_returns) ** 2
                        value_clipped = mb_returns + torch.clamp(
                            new_values - mb_returns,
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        value_loss_clipped = (value_clipped - mb_returns) ** 2
                        value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
                        value_loss = 0.5 * value_loss_max.mean()
                    else:
                        value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                    # total loss
                    entropy = new_entropys.mean()
                    loss = policy_loss - self.ent_coef * entropy + self.vf_coef * value_loss

                    # backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    for param_group in self.optimizer.param_groups:
                        nn.utils.clip_grad_norm_(param_group["params"], self.max_grad_norm)
                    self.optimizer.step()

            # Log reward and loss
            total_reward = np.sum(rewards)
            self.reward_history.append(total_reward)
            self.value_loss_history.append(value_loss.detach().numpy())
            self.policy_loss_history.append(policy_loss.detach().numpy())
            self.current_iteration += 1
            
            # # Save model every 100 iterations
            # if self.current_iteration % 100 == 0:
            #     self.save(self.save_path)
            
            if self.verbose:
                print(f"Epoch: [{self.current_iteration}/{self.start_iteration+self.max_iterations}], total_reward: {total_reward:.2f}")
                print(f" - total loss: {loss:.2f}, policy loss: {policy_loss:.2f}, value loss: {value_loss:.2f}, entropy: {entropy:.2f}")

        if self.verbose:
            print("PPO Training finished!")

    def compute_action(self, current_state: np.ndarray, current_time: Optional[float] = None) -> np.ndarray:

        state_tensor = torch.tensor(current_state, dtype=torch.float32)

        # sampling or simply use the mean value from actor net
        #mean = self.policy_net(state_tensor)
        #std = torch.exp(self.log_std)
        #dist = torch.distributions.Normal(mean, std)
        action_unsquashed = self.policy_net(state_tensor)
        action_squashed = torch.tanh(action_unsquashed)
        action = self._action_reverse_tf(action_squashed.detach().numpy())

        return action
    
    def plot_policy_heatmap(self, x_range=(-1.5, 1.5), y_range=(-1.0, 1.0), resolution=50):

        x_vals = torch.linspace(x_range[0], x_range[1], resolution)
        y_vals = torch.linspace(y_range[0], y_range[1], resolution)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing="xy")
        
        states = torch.stack([X.flatten(), Y.flatten()], dim=-1) 

        with torch.no_grad():
            actions_unsquashed = self.policy_net(states)
            actions_squashed = torch.tanh(actions_unsquashed).detach().numpy()

        actions_squashed = actions_squashed.reshape(resolution, resolution) 
        actions = self._action_reverse_tf(actions_squashed)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(actions, xticklabels=False, yticklabels=False, cmap="coolwarm", cbar=True, cbar_kws={'label': 'Acceleration [m/s^2]'})
        plt.title("Policy Heatmap")
        plt.xlabel("Position [m]")
        plt.ylabel("Velocity [m/s]")
        plt.show()

    def plot_training_curve(self):
        rewards = np.array(self.reward_history)
        value_loss = np.array(self.value_loss_history)
        policy_loss = np.array(self.policy_loss_history)
        steps = np.linspace(0, self.start_iteration+self.max_iterations, len(rewards))
        window_size = 50

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))  

        # Step 1: Reward curve
        if rewards.ndim == 1:  
            axes[0].plot(steps, rewards, color='blue', label='max', linewidth=2)
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            axes[0].plot(steps[:len(smoothed_rewards)], smoothed_rewards, color='cyan', label='smoothed', linewidth=2)
        else: 
            mean_reward = np.mean(rewards, axis=0)
            max_reward = np.max(rewards, axis=0)
            min_reward = np.min(rewards, axis=0)
            axes[0].plot(steps, mean_reward, color='blue', label='mean', linewidth=2)
            axes[0].plot(steps, max_reward, color='red', label='max', linewidth=2)
            axes[0].plot(steps, min_reward, color='black', label='min', linewidth=2)
            axes[0].fill_between(steps, max_reward, min_reward, color='red', alpha=0.2)
            smoothed_mean_reward = np.convolve(mean_reward, np.ones(window_size)/window_size, mode='valid')
            axes[0].plot(steps[:len(smoothed_mean_reward)], smoothed_mean_reward, color='cyan', label='smoothed', linewidth=2)
        
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('PPO Reward Curve')
        axes[0].legend()
        axes[0].grid(True)
        
        # Step 2: Value loss curve
        smoothed_value_loss = np.convolve(value_loss, np.ones(window_size)/window_size, mode='valid')
        axes[1].plot(steps, value_loss, color='green', label='Value Loss', linewidth=2)
        axes[1].plot(steps[:len(smoothed_value_loss)], smoothed_value_loss, color='lime', label='smoothed', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Value Loss Curve')
        axes[1].legend()
        axes[1].grid(True)
        
        # Step 3: Policy loss curve
        smoothed_policy_loss = np.convolve(policy_loss, np.ones(window_size)/window_size, mode='valid')
        axes[2].plot(steps, policy_loss, color='orange', label='Policy Loss', linewidth=2)
        axes[2].plot(steps[:len(smoothed_policy_loss)], smoothed_policy_loss, color='gold', label='smoothed', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Policy Loss Curve')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()  
        plt.show()