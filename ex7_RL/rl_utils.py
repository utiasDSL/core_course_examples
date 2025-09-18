import numpy as np
import sympy as sp
import pickle
import casadi as ca
import scipy.linalg
import copy
import time
import os
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from abc import ABC, abstractmethod
from typing import List, Callable, Union, Optional, Tuple, Any
from functools import wraps
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from IPython.display import display, HTML
from scipy.interpolate import interp2d
from cvxopt import matrix, solvers
import pytope as pt
from scipy.spatial import ConvexHull, QhullError, cKDTree

from utils.env import Env, Dynamics, Env_rl_d
from utils.controller import BaseController, check_input_constraints, timing






'''------Model-based RL Controller------'''

# Derived class for model-based RL Controller, solved with General Policy Iteration
class GPIController(BaseController):
    def __init__(self, 
                 mdp: Env_rl_d, 
                 freq: float, 
                 gamma: float = 0.95,
                 precision_pe: float = 1e-6,
                 precision_pi: float = 1e-6,
                 max_ite_pe: int = 50,
                 max_ite_pi: int = 100,
                 name: str = 'GPI', 
                 type: str = 'RL', 
                 verbose: bool = True
                 ) -> None:
        
        super().__init__(mdp, mdp.dynamics, freq, name, type, verbose)
        
        # VI: max_ite_pi = 1, max_ite_pe > 500, precision_pi not needed, reasonable precision_pe (1e-6)
        # PI: max_ite_pi > 50, max_ite_pe > 100, reasonable precision_pi (1e-6), precision_pe not needed
        self.precision_pe = precision_pe
        self.precision_pi = precision_pi
        self.max_ite_pe = max_ite_pe
        self.max_ite_pi = max_ite_pi

        self.mdp = mdp
        self.gamma = gamma

        self.dim_states = self.mdp.num_states
        self.dim_inputs = self.mdp.num_actions

        self.init_state = self.env.init_state
        self.target_state = self.env.target_state

        # Initialize policy and value function
        self.policy = np.zeros(self.dim_states, dtype=int)  # Initial policy: choose action 0 for all states
        self.value_function = np.zeros(self.dim_states)  # Initial value function

        self.num_pe_iterations = 0
        self.num_pi_iterations = 0
    
    @timing
    def setup(self) -> None:
        """Perform GPI to compute the optimal policy."""

        new_value_function = np.zeros_like(self.value_function)

        for pi_iteration in range(self.max_ite_pi):

            # Policy Evaluation
            new_value_function, pe_iteration = self.policy_evaluation(new_value_function)

            # Policy Improvement
            policy_stable, old_policy = self.policy_improvement()

            # Check for convergence in policy improvement
            if policy_stable and np.max(np.abs(new_value_function - self.value_function)) < self.precision_pi:
                if self.verbose:
                    print(f"Policy converged after {pi_iteration + 1} iterations.")
                    print(f"Optimal Policy: {self.policy}")
                break
            elif self.verbose:
                print(f"Policy improvement iteration {pi_iteration + 1}, still not converged, keep running.")
                print(f"Last Policy: {old_policy}")
                print(f"Current Policy: {self.policy}")

            self.num_pi_iterations += 1

        print(f"Total policy evaluation iterations: {self.num_pe_iterations}")
        print(f"Total policy improvement iterations: {self.num_pi_iterations}")

    def policy_evaluation(
        self, 
        new_value_function: np.ndarray
    ) -> tuple:
        
        # Policy Evaluation
        for pe_iteration in range(self.max_ite_pe):

            self.value_function = copy.copy(new_value_function)

            new_value_function = np.zeros_like(self.value_function)
            
            for state_index in range(self.dim_states):

                action_index = self.policy[state_index]

                for next_state_index in range(self.dim_states):

                    new_value_function[state_index] += self.mdp.T[action_index][state_index, next_state_index] * (
                        self.mdp.R[action_index][state_index, next_state_index] + self.gamma * self.value_function[next_state_index]
                    )

            print(f"self.value_function: {self.value_function}")
            print(f"new_value_function: {new_value_function}")

            self.num_pe_iterations += 1

            # Check for convergence in policy evaluation
            if np.max(np.abs(new_value_function - self.value_function)) < self.precision_pe:
                if self.verbose:
                    print(f"Policy evaluation converged after {pe_iteration + 1} iterations, max error: {np.max(np.abs(new_value_function - self.value_function))}.")
                break
            elif self.verbose:
                print(f"Policy evaluation iteration {pe_iteration + 1}, max error: {np.max(np.abs(new_value_function - self.value_function))}")
        
        self.value_function = copy.copy(new_value_function)

        return new_value_function, pe_iteration

    def policy_improvement(self):

        # Policy Improvement
        policy_stable = True
        old_policy = copy.copy(self.policy)
        for state_index in range(self.dim_states):
            
            q_values = np.zeros(self.dim_inputs)

            # Compute Q-values for all actions
            for action_index in range(self.dim_inputs):
                for next_state_index in range(self.dim_states):

                    q_values[action_index] += self.mdp.T[action_index][state_index, next_state_index] * (
                        self.mdp.R[action_index][state_index, next_state_index] + self.gamma * self.value_function[next_state_index]
                    )

            # Update policy greedily
            self.policy[state_index] = np.argmax(q_values)

            # Check for convergence in policy improvement
            if old_policy[state_index] != self.policy[state_index]:
                policy_stable = False

        return policy_stable, old_policy

    @check_input_constraints
    def compute_action(
        self, 
        current_state: np.ndarray, 
        current_time: int
    ) -> np.ndarray:
        """
        Use the optimal policy to compute the action for the given state.
        """
        # Find the nearest discrete state index
        state_index = self.mdp.nearest_state_index_lookup(current_state)
        
        # Get the optimal action from the policy
        action_index = self.policy[state_index]
        action = self.mdp.input_space[action_index]

        return np.array([action])

    def plot_heatmaps(self):
        """
        Visualize the policy and cost as a 2D state-value map.
        """
        value_map = self.value_function.reshape(self.mdp.num_pos, self.mdp.num_vel)
        policy_map = self.policy.reshape(self.mdp.num_pos, self.mdp.num_vel)
        # turn policy map into input map
        input_map = self.mdp.input_space[self.policy].reshape(self.mdp.num_pos, self.mdp.num_vel)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Plot policy (U)
        im1 = axs[0].imshow(input_map, extent=[
            self.mdp.pos_partitions[0], self.mdp.pos_partitions[-1],
            self.mdp.vel_partitions[0], self.mdp.vel_partitions[-1]
        ], origin='lower', aspect='auto', cmap='viridis')
        axs[0].set_title('Optimal Policy')
        axs[0].set_xlabel('Car Position')
        axs[0].set_ylabel('Car Velocity')
        fig.colorbar(im1, ax=axs[0], orientation='vertical')

        # Plot cost-to-go (J)
        im2 = axs[1].imshow(value_map, extent=[
            self.mdp.pos_partitions[0], self.mdp.pos_partitions[-1],
            self.mdp.vel_partitions[0], self.mdp.vel_partitions[-1]
        ], origin='lower', aspect='auto', cmap='viridis')
        axs[1].set_title('Optimal Cost')
        axs[1].set_xlabel('Car Position')
        axs[1].set_ylabel('Car Velocity')
        fig.colorbar(im2, ax=axs[1], orientation='vertical')

        plt.tight_layout()
        plt.show()






'''------Model-free RL Controllers------'''

# Derived class for model-free RL Controller, solved with Monte-Carlo method
class MCRLController(BaseController):
    def __init__(self, 
                 mdp: Env_rl_d, 
                 freq: float, 
                 epsilon: float = 0.3, 
                 k_epsilon: float = 0.99, 
                 epsilon_min: float = 0.01,
                 learning_rate: float = 0.2, 
                 gamma: float = 0.95,
                 max_iterations: int = 5000, 
                 max_steps_per_episode: int = 500, 
                 name: str = 'MCRL', 
                 type: str = 'RL', 
                 verbose: bool = True
                 ) -> None:
        
        super().__init__(mdp, mdp.dynamics, freq, name, type, verbose)

        self.epsilon = epsilon  # exploration rate
        self.k_epsilon = k_epsilon  # decay factor for exploration rate
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.max_steps_per_episode = max_steps_per_episode

        self.mdp = mdp

        self.dim_states = self.mdp.num_states
        self.dim_inputs = self.mdp.num_actions

        self.init_state = self.env.init_state
        self.target_state = self.env.target_state

        # Initialize Q table with all 0s
        self.Q = np.zeros((self.dim_states, self.dim_inputs)) 
        self.state_action_counts = np.zeros((self.dim_states, self.dim_inputs))
        
        # Initialize policy as None
        self.policy = np.zeros((self.dim_states)) 
        self.value_function = np.zeros((self.dim_states)) 

        # For training curve plotting
        self.residual_rewards = []
        self.residual_rewards_smoothed = None
        self.step_list = []
        self.epsilon_list = []
        self.SR_100epsd = [] # Successful rounds
        self.F_100epsd = [] # Failure rounds
        self.TO_100epsd = [] # Time out rounds

    def _get_action_probabilities(
        self, 
        state_index: int
    ) -> np.ndarray:
        
        """Calculate the action probabilities using epsilon-soft policy."""

        probabilities = np.ones(self.dim_inputs) * (self.epsilon / self.dim_inputs)
        best_action = np.argmax(self.Q[state_index, :])
        probabilities[best_action] += (1.0 - self.epsilon)

        return probabilities

    def setup(self) -> None:

        for iteration in range(self.max_iterations):

            episode = []  # storage state, action and reward for current episode
            total_reward = 0  # total reward for current episode
            total_steps = 0  # total steps for current episode

            if iteration % 100 == 0:

                if iteration != 0:
                    # Record the SR_100epsd, F_100epsd, TO_100epsd
                    self.SR_100epsd.append(SR_100epsd)
                    self.F_100epsd.append(F_100epsd)
                    self.TO_100epsd.append(TO_100epsd)
                
                SR_100epsd = 0
                F_100epsd = 0
                TO_100epsd = 0

            # Start from init state
            # Note: Here we restrict initial state distribution to boost the training but one can also 
            #       randomly initialize state given sufficient interaction to improve the generalization.
            current_state = self.mdp.init_state
            current_state_index = self.mdp.nearest_state_index_lookup(current_state)
            # Randomly choose a state to start
            #current_state_index = np.random.choice(self.dim_states)
            #current_state = self.mdp.state_space[:, current_state_index]
            # Randomly choose a position with 0 vel to start
            #current_pos_index = np.random.choice(self.mdp.num_pos)
            #current_state = np.array([self.mdp.pos_partitions[current_pos_index], 0.0])
            #current_state_index = self.mdp.nearest_state_index_lookup(current_state)
            
            # Generate an episode
            for step in range(self.max_steps_per_episode):
                # Choose action based on epsilon-soft policy
                action_probabilities = self._get_action_probabilities(current_state_index)
                action_index = np.random.choice(np.arange(self.dim_inputs), p=action_probabilities)
                current_input = self.mdp.input_space[action_index]

                # Take action and observe the next state and reward
                next_state, reward = self.mdp.one_step_forward(current_state, current_input)
                next_state_index = self.mdp.nearest_state_index_lookup(next_state)
                total_reward += reward
                total_steps += 1 

                # Store the state, action and reward for this step
                episode.append((current_state_index, action_index, reward))

                # Check if the episode is finished
                terminate_condition_1 = next_state[0]>self.mdp.pos_partitions[-1]
                terminate_condition_2 = next_state[0]<self.mdp.pos_partitions[0]
                terminate_condition_3 = np.all(self.mdp.state_space[:, next_state_index]==self.target_state)
                
                if terminate_condition_1 or terminate_condition_2 or terminate_condition_3:
                    if terminate_condition_3:
                        SR_100epsd += 1
                        if self.verbose:
                            print(f"Iteration {iteration + 1}/{self.max_iterations}: finished successfully at step {step}! epsilon: {self.epsilon:.4f}, residual reward: {total_reward:.2f}")
                    else:
                        F_100epsd += 1
                        if self.verbose:
                            print(f"Iteration {iteration + 1}/{self.max_iterations}: episode failed at step {step}! epsilon: {self.epsilon:.4f}, residual reward: {total_reward:.2f}")
                    break

                if step == self.max_steps_per_episode-1:
                    TO_100epsd +=1
                    if self.verbose:
                        print(f"Iteration {iteration + 1}/{self.max_iterations}: time out (step: {step})! epsilon: {self.epsilon:.4f}, residual reward: {total_reward:.2f}")
                    
                # Move to the next state
                current_state_index = next_state_index
                current_state = self.mdp.state_space[:, current_state_index]

            # Update Q table using Monte Carlo method
            G = 0  # Return
            for state_index, action_index, reward in reversed(episode):
                # Cumulative return
                G = reward + self.gamma * G
                
                # Factor in recursive estimation
                self.state_action_counts[state_index, action_index] += 1
                alpha = self.learning_rate
                #alpha = 1.0 / self.state_action_counts[state_index, action_index]

                # Update Q using MC and log TD error
                td_error = G - self.Q[state_index, action_index]
                self.Q[state_index, action_index] += alpha * td_error

            # Decrease epsilon
            self.epsilon *= self.k_epsilon
            self.epsilon = max(self.epsilon, self.epsilon_min)
            self.epsilon_list.append(self.epsilon)

            # Record the residual reward and loss
            self.residual_rewards.append(total_reward)
            self.step_list.append(total_steps)

        # Return the deterministic policy and value function
        self.policy = np.argmax(self.Q, axis=1)
        self.value_function = np.max(self.Q, axis=1)

        # Repeat success/failure stats for plotting
        self.SR_100epsd = np.repeat(self.SR_100epsd, 100)/100
        self.F_100epsd = np.repeat(self.F_100epsd, 100)/100
        self.TO_100epsd = np.repeat(self.TO_100epsd, 100)/100

        if self.verbose:
            print("Training finished！")

    @check_input_constraints
    def compute_action(
        self, 
        current_state: np.ndarray, 
        current_time: int
    ) -> np.ndarray:
        
        """
        Use the optimal policy to compute the action for the given state.
        """
        # Find the nearest discrete state index
        state_index = self.mdp.nearest_state_index_lookup(current_state)
        
        # Get the optimal action from the policy
        action_index = self.policy[state_index]
        action = self.mdp.input_space[action_index]

        return np.array([action])
    
    def postprocessing(
        self, 
        window: int = 20
    ) -> tuple:

        def moving_average(data, window=20):
            """Return moving average and std of a 1D array"""
            data = np.array(data)
            if window <= 1:
                return data.copy()
            
            kernel = np.ones(window)
            z = np.ones(len(data))        
            smoothed = np.convolve(data, kernel, mode='same') / np.convolve(z, kernel, mode='same')
            return smoothed
        
        # Smooth reward curve
        self.residual_rewards_smoothed = moving_average(self.residual_rewards, window)

        return self.residual_rewards_smoothed, self.SR_100epsd
        
    def plot_training_curve(
        self, 
        title: str = None
    ) -> None:
        
        """
        Visualize the training curve of residual rewards with smoothed version and confidence band,
        and holdsignal for sr_100epsd in 2 other figures.
        """

        fig, axs = plt.subplots(1, 3, figsize=(12, 3))

        # Plot residual rewards + smooth + confidence band
        if self.residual_rewards_smoothed is not None and len(self.residual_rewards_smoothed):
            axs[0].plot(self.residual_rewards, alpha=0.3, label="Raw reward")
            axs[0].plot(self.residual_rewards_smoothed, label="Smoothed", color="C0")
        else:
            axs[0].plot(self.residual_rewards, label="Raw reward")
        axs[0].set_title('Total Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')
        axs[0].legend()

        # Plot success/fail/time-out rounds
        axs[1].plot(self.SR_100epsd, label='Success rounds')
        axs[1].plot(self.F_100epsd, label='Fail rounds')
        axs[1].plot(self.TO_100epsd, label='Time out')
        axs[1].set_title('Statistics per 100 Episodes')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Percentage of Trails')
        axs[1].set_ylim(0, 1)
        axs[1].legend()

        # Plot epsilon
        axs[-1].plot(self.epsilon_list)
        axs[-1].set_title('Epsilon')
        axs[-1].set_xlabel('Episode')
        axs[-1].set_ylabel('Epsilon')
        axs[-1].set_ylim(0, 1)

        if title is not None:
            fig.suptitle(title, fontsize=12, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.show()


# Derived class for model-free RL Controller, solved with Q-learning algorithm
class QLearningController(BaseController):
    def __init__(self, 
                 mdp: Env_rl_d, 
                 freq: float, 
                 epsilon: float = 0.3, 
                 k_epsilon: float = 0.99, 
                 epsilon_min: float = 0.01,
                 learning_rate: float = 0.2, 
                 gamma: float = 0.95,
                 max_iterations: int = 5000, 
                 max_steps_per_episode: int = 500, 
                 name: str = 'Q-learning', 
                 type: str = 'RL', 
                 verbose: bool = True
                 ) -> None:
        
        super().__init__(mdp, mdp.dynamics, freq, name, type, verbose)

        self.epsilon = epsilon  # exploration rate
        self.k_epsilon = k_epsilon  # decay factor for exploration rate
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.max_steps_per_episode = max_steps_per_episode

        self.mdp = mdp

        self.dim_states = self.mdp.num_states
        self.dim_inputs = self.mdp.num_actions

        self.init_state = self.env.init_state
        self.target_state = self.env.target_state

        # Initialize Q table with all 0s
        self.Q = np.zeros((self.dim_states, self.dim_inputs)) 
        self.state_action_counts = np.zeros((self.dim_states, self.dim_inputs))
        
        # Initialize policy as None
        self.policy = np.zeros((self.dim_states)) 
        self.value_function = np.zeros((self.dim_states)) 

        # For training curve plotting
        self.residual_rewards = []
        self.residual_rewards_smoothed = None
        self.step_list = []
        self.epsilon_list = []
        self.SR_100epsd = [] # Successful rounds
        self.F_100epsd = [] # Failure rounds
        self.TO_100epsd = [] # Time out rounds

    def _get_action_probabilities(
        self, 
        state_index: int
    ) -> np.ndarray:
        
        """Calculate the action probabilities using epsilon-soft policy."""

        probabilities = np.ones(self.dim_inputs) * (self.epsilon / self.dim_inputs)
        best_action = np.argmax(self.Q[state_index])
        probabilities[best_action] += (1.0 - self.epsilon)

        return probabilities

    def setup(self) -> None:

        for iteration in range(self.max_iterations):

            total_reward = 0  # To accumulate rewards for this episode
            total_steps = 0  # To count total steps in this episode

            if iteration % 100 == 0:

                if iteration != 0:
                    # Record the SR_100epsd, F_100epsd, TO_100epsd
                    self.SR_100epsd.append(SR_100epsd)
                    self.F_100epsd.append(F_100epsd)
                    self.TO_100epsd.append(TO_100epsd)
                
                SR_100epsd = 0
                F_100epsd = 0
                TO_100epsd = 0

            # Start from init state
            # Note: Here we restrict initial state distribution to boost the training but one can also 
            #       randomly initialize state given sufficient interaction to improve the generalization.
            current_state = self.mdp.init_state
            current_state_index = self.mdp.nearest_state_index_lookup(current_state)
            # Randomly choose a state to start
            #current_state_index = np.random.choice(self.dim_states)
            #current_state = self.mdp.state_space[:, current_state_index]
            # Randomly choose a position with 0 vel to start
            #current_pos_index = np.random.choice(self.mdp.num_pos)
            #current_state = np.array([self.mdp.pos_partitions[current_pos_index], 0.0])
            #current_state_index = self.mdp.nearest_state_index_lookup(current_state)

            for step in range(self.max_steps_per_episode):

                # Choose action based on epsilon-soft policy
                action_probabilities = self._get_action_probabilities(current_state_index)
                action_index = np.random.choice(np.arange(self.dim_inputs), p=action_probabilities)
                current_input = self.mdp.input_space[action_index]

                # Take action and observe the next state and reward
                next_state, reward = self.mdp.one_step_forward(current_state, current_input)
                next_state_index = self.mdp.nearest_state_index_lookup(next_state)
                total_reward += reward  # Accumulate total reward
                total_steps += 1  # Increment step count
                
                # Update Q table and compute TD error
                td_error = reward + self.gamma * np.max(self.Q[next_state_index, :]) - self.Q[current_state_index, action_index]

                # Factor in recursive estimation
                self.state_action_counts[current_state_index, action_index] += 1
                alpha = self.learning_rate
                #alpha = 1.0 / self.state_action_counts[current_state_index, action_index]

                # Update Q table and log TD error
                self.Q[current_state_index, action_index] += alpha * td_error
                
                # Check if the episode is finished
                terminate_condition_1 = next_state[0]>self.mdp.pos_partitions[-1]
                terminate_condition_2 = next_state[0]<self.mdp.pos_partitions[0]
                terminate_condition_3 = np.all(self.mdp.state_space[:, next_state_index]==self.target_state)

                if terminate_condition_1 or terminate_condition_2 or terminate_condition_3:
                    if terminate_condition_3:
                        SR_100epsd += 1
                        if self.verbose:
                            print(f"Iteration {iteration + 1}/{self.max_iterations}: finished successfully at step {step}! epsilon: {self.epsilon:.4f}, residual reward: {total_reward:.2f}")
                    else:
                        F_100epsd += 1
                        if self.verbose:
                            print(f"Iteration {iteration + 1}/{self.max_iterations}: episode failed at step {step}! epsilon: {self.epsilon:.4f}, residual reward: {total_reward:.2f}")
                    break

                else:
                    
                    # Move to the next state
                    current_state_index = next_state_index
                    current_state = self.mdp.state_space[:, current_state_index]
                
                if step == self.max_steps_per_episode-1:
                    TO_100epsd +=1
                    if self.verbose:
                        print(f"Iteration {iteration + 1}/{self.max_iterations}: time out (step: {step})! epsilon: {self.epsilon:.4f}, residual reward: {total_reward:.2f}")
                    

            # Decrease epsilon
            self.epsilon *= self.k_epsilon
            self.epsilon = max(self.epsilon, self.epsilon_min)
            self.epsilon_list.append(self.epsilon)

            # Record the residual reward and loss
            self.residual_rewards.append(total_reward)
            self.step_list.append(total_steps)
        
        # Return the deterministic policy and value function
        self.policy = np.argmax(self.Q, axis=1)
        self.value_function = np.max(self.Q, axis=1)

        # Repeat success/failure stats for plotting
        self.SR_100epsd = np.repeat(self.SR_100epsd, 100)/100
        self.F_100epsd = np.repeat(self.F_100epsd, 100)/100
        self.TO_100epsd = np.repeat(self.TO_100epsd, 100)/100
        
        if self.verbose:
            print("Training finished！")

    @check_input_constraints
    def compute_action(
        self, 
        current_state: np.ndarray, 
        current_time: int
    ) -> np.ndarray:
        
        """
        Use the optimal policy to compute the action for the given state.
        """
        # Find the nearest discrete state index
        state_index = self.mdp.nearest_state_index_lookup(current_state)
        
        # Get the optimal action from the policy
        action_index = self.policy[state_index]
        action = self.mdp.input_space[action_index]

        return np.array([action])

    def postprocessing(
        self, 
        window: int = 20
    ) -> tuple:

        def moving_average(data, window=20):
            """Return moving average and std of a 1D array"""
            data = np.array(data)
            if window <= 1:
                return data.copy()
            
            kernel = np.ones(window)
            z = np.ones(len(data))        
            smoothed = np.convolve(data, kernel, mode='same') / np.convolve(z, kernel, mode='same')
            return smoothed
        
        # Smooth reward curve
        self.residual_rewards_smoothed = moving_average(self.residual_rewards, window)

        return self.residual_rewards_smoothed, self.SR_100epsd
        
    def plot_training_curve(
        self, 
        title: str = None
    ) -> None:
        
        """
        Visualize the training curve of residual rewards with smoothed version and confidence band,
        and holdsignal for sr_100epsd in 2 other figures.
        """

        fig, axs = plt.subplots(1, 3, figsize=(12, 3))

        # Plot residual rewards + smooth + confidence band
        if self.residual_rewards_smoothed is not None and len(self.residual_rewards_smoothed):
            axs[0].plot(self.residual_rewards, alpha=0.3, label="Raw reward")
            axs[0].plot(self.residual_rewards_smoothed, label="Smoothed", color="C0")
        else:
            axs[0].plot(self.residual_rewards, label="Raw reward")
        axs[0].set_title('Total Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')
        axs[0].legend()

        # Plot success/fail/time-out rounds
        axs[1].plot(self.SR_100epsd, label='Success rounds')
        axs[1].plot(self.F_100epsd, label='Fail rounds')
        axs[1].plot(self.TO_100epsd, label='Time out')
        axs[1].set_title('Statistics per 100 Episodes')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Percentage of Trails')
        axs[1].set_ylim(0, 1)
        axs[1].legend()

        # Plot epsilon
        axs[-1].plot(self.epsilon_list)
        axs[-1].set_title('Epsilon')
        axs[-1].set_xlabel('Episode')
        axs[-1].set_ylabel('Epsilon')
        axs[-1].set_ylim(0, 1)

        if title is not None:
            fig.suptitle(title, fontsize=12, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.show()






'''------Utils------'''

# Experiment runner for RL controllers
class RLExperimentRunner:
    def __init__(
        self, 
        controller_instances: dict, 
        seed_list: list, 
        save_dir: str
    ) -> None:
        
        """
        controller_instances: dict mapping controller name to a single controller instance (template)
        seed_list: list of seeds to use for all controllers
        save_dir: directory to save results
        """
        self.controller_instances = controller_instances  # e.g., {"mcrl": controller_obj, ...}
        self.seed_list = seed_list
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def run_all(self):
        for name, controller_template in self.controller_instances.items():
            for seed in self.seed_list:
                save_path = os.path.join(self.save_dir, f"{name}_seed{seed}.npz")
                if os.path.exists(save_path):
                    print(f"[Skipped] {save_path} already exists.")
                    continue

                np.random.seed(seed)
                controller_copy = copy.deepcopy(controller_template)
                controller_copy.setup()
                reward,  SR = controller_copy.postprocessing(window=50)

                np.savez(save_path, reward=reward, success_rate=SR, policy=controller_copy.policy)
                print(f"[Saved] {save_path}")

    def load_results(self):
        result_dict = {}
        for name in self.controller_instances.keys():
            all_rewards = []
            all_success_rates = []
            for seed in self.seed_list:
                file_path = os.path.join(self.save_dir, f"{name}_seed{seed}.npz")
                if not os.path.exists(file_path):
                    print(f"[Warning] Missing file: {file_path}, skipping.")
                    continue
                data = np.load(file_path)
                all_rewards.append(data['reward'])
                all_success_rates.append(data['success_rate'])
            result_dict[name] = {
                'reward': np.vstack(all_rewards),
                'success_rate': np.vstack(all_success_rates)
            }
        return result_dict
    
    def get_trained_controller(
        self, 
        name: str, 
        seed: int
    ) -> BaseController:
        
        """
        Return a deepcopy of the template controller whose .policy has been
        replaced by the stored one for (name, seed). If the file is absent,
        raise FileNotFoundError.
        """
        f = os.path.join(self.save_dir, f"{name}_seed{seed}.npz")
        if not os.path.exists(f):
            raise FileNotFoundError(f"No saved data for {name}, seed {seed}")

        data = np.load(f, allow_pickle=True)
        policy_np = data["policy"]

        ctrl = copy.deepcopy(self.controller_instances[name])
        ctrl.policy = policy_np
        return ctrl

    def compute_statistics(self):
        results = self.load_results()
        stats = {}
        for name, data in results.items():
            reward = data['reward']
            success = data['success_rate']
            stats[name] = {
                'reward_mean': np.mean(reward, axis=0),
                'reward_std': np.std(reward, axis=0),
                'reward_mean_min': np.min(reward, axis=0),
                'reward_mean_max': np.max(reward, axis=0),
                'success_mean': np.mean(success, axis=0),
                'success_std': np.std(success, axis=0),
                'success_mean_min': np.min(success, axis=0),
                'success_mean_max': np.max(success, axis=0)
            }
        return stats

    def plot(
        self, 
        title: str = None
    ) -> None:

        stats = self.compute_statistics()
        # x‑axes per metric can differ in length – compute individually later
        fig, (ax_r, ax_sr) = plt.subplots(1, 2, figsize=(14, 5))

        # ------- Reward subplot --------
        for name, val in stats.items():
            x_r = np.arange(len(val['reward_mean']))
            ax_r.plot(x_r, val['reward_mean'], label=f"{name}")
            if len(self.seed_list) > 1:
                ax_r.fill_between(x_r,
                                val['reward_mean'] - 3 * val['reward_std'],
                                val['reward_mean'] + 3 * val['reward_std'], alpha=0.3)
                                #val['reward_mean_min'],
                                #val['reward_mean_max'], alpha=0.3)
        if len(self.seed_list) > 1:
            ax_r.set_title("Reward Curve: Mean ± 3 * Std")
        else:
            ax_r.set_title("Reward Curve")
        ax_r.set_xlabel("Episode")
        ax_r.set_ylabel("Reward")
        ax_r.grid(True)
        ax_r.legend()

        # ------- Success‑rate subplot --------
        for name, val in stats.items():
            x_sr = np.arange(len(val['success_mean']))
            ax_sr.plot(x_sr, val['success_mean'], label=f"{name}")
            if len(self.seed_list) > 1:
                ax_sr.fill_between(x_sr,
                                val['success_mean'] - 3 * val['success_std'],
                                val['success_mean'] + 3 * val['success_std'], alpha=0.3)
                                #val['success_mean_min'],
                                #val['success_mean_max'], alpha=0.3)
        if len(self.seed_list) > 1:
            ax_sr.set_title("Success Rate: Mean ± 3 * Std")
        else:
            ax_sr.set_title("Success Rate")
        ax_sr.set_xlabel("Episode")
        ax_sr.set_ylabel("Success Rate")
        ax_sr.grid(True)
        ax_sr.legend()

        if title is not None:
            fig.suptitle(title, fontsize=20, fontweight='bold', y=1.05)

        plt.tight_layout()
        plt.show()