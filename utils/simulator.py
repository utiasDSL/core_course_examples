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

from utils.env import Env, Dynamics
from utils.controller import BaseController







class Simulator:
    def __init__(
        self,
        dynamics: Dynamics = None,
        controller: BaseController = None,
        env: Env = None,
        dt: float = None,
        t_terminal: float = None,
        verbose: bool = False
    ) -> None:
        
        self.dynamics = dynamics
        self.controller = controller
        self.env = env

        self.verbose = verbose

        # Initialize attributes only if 'env' is provided
        if env is not None:
            self.init_state = env.init_state
        else:
            self.init_state = None

        # Initialize timeline attributes only if 'dt' and 't_terminal' are provided
        if dt is not None and t_terminal is not None:
            self.t_0 = 0
            self.t_terminal = t_terminal
            self.dt = dt
            self.t_eval = np.linspace(
                self.t_0, self.t_terminal, int((self.t_terminal - self.t_0) / self.dt) + 1
            )
        else:
            self.t_0 = None
            self.t_terminal = None
            self.dt = None
            self.t_eval = None

        # Initialize recording lists
        self.state_traj = []
        self.input_traj = []
        self.nominal_input_traj = []
        self.state_pred_traj = []
        self.input_pred_traj = []
        self.cost2go_arr = None
        self.counter = 0

        # Set controller character if controller is provided
        self.controller_name = controller.name if controller is not None else None
        self.controller_type = controller.type if controller is not None else None

    def reset_counter(self) -> None:
        self.counter = 0
    
    def run_simulation(self) -> None:
        
        # Initialize state vector
        current_state = self.init_state
        self.state_traj.append(current_state)

        for current_time in self.t_eval[:-1]:

            # In RL case, once the target is reached, stop the car at target position forever
            if self.controller.type == 'RL':
                # Discrete state RL
                if hasattr(self.controller.mdp, 'num_pos'):
                    pos_bin = (self.controller.mdp.pos_ubs-self.controller.mdp.pos_lbs)/self.controller.mdp.num_pos
                # Continuous state RL (DRL)
                else:
                    pos_bin = (self.controller.mdp.pos_ubs-self.controller.mdp.pos_lbs)/30
                if np.linalg.norm(current_state[0]-self.env.target_position) < pos_bin/2:
                    current_input = 0.0
                    current_state = self.env.target_state
                else:
                    input_cmd = self.controller.compute_action(current_state, self.counter)
                    current_input = input_cmd
                    current_state = self.dynamics.one_step_forward(current_state, current_input, self.dt)
            else:
                # Get current state, and call controller to calculate input
                if self.controller.type == 'MPC':
                    input_cmd, state_pred, input_pred = self.controller.compute_action(current_state, self.counter)
                    # Log the predictions
                    self.state_pred_traj.append(state_pred)
                    self.input_pred_traj.append(input_pred)
                elif self.controller.type == 'RMPC':
                    input_cmd, state_pred, input_pred, nominal_input = self.controller.compute_action(current_state, self.counter)
                    # Log the predictions
                    self.state_pred_traj.append(state_pred)
                    self.input_pred_traj.append(input_pred)
                else:
                    input_cmd = self.controller.compute_action(current_state, self.counter)

                current_input = input_cmd
                current_state = self.dynamics.one_step_forward(current_state, current_input, self.dt)
            #print(f"sim_state:{current_state}")

            # Log the results
            self.state_traj.append(current_state)
            self.input_traj.append(np.array(current_input).flatten())
            if self.controller.type == 'RMPC':
                self.nominal_input_traj.append(np.array(nominal_input).flatten())

            # Update timer
            self.counter += 1
        
        print("Simulation finished, will start plotting")
        self.reset_counter()
    
    def save(self, filename='dp_sim.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({
                'controller_name': self.controller_name,
                'controller_type': self.controller_type,
                'env': self.env,
                't_eval': self.t_eval,
                'state_traj': self.state_traj,
                'input_traj': self.input_traj,
            }, f)

    def load(self, filename='dp_sim.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

            self.controller_name = data['controller_name']
            self.controller_type = data['controller_type']

            self.env = data['env']
        
            self.t_eval = data['t_eval']
            self.t_0 = self.t_eval[0]
            self.t_terminal = self.t_eval[-1]
            self.dt = self.t_eval[1] - self.t_eval[0]
            
            self.state_traj = data['state_traj']
            self.input_traj = data['input_traj']
            self.init_state = data['state_traj'][0]

    def get_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Get state and input trajectories, return in ndarray form '''

        # Transform state and input traj from list to ndarray
        state_traj = np.array(self.state_traj)
        input_traj = np.array(self.input_traj)

        return state_traj, input_traj

    def compute_cost2go(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        target_state: np.ndarray
    ) -> float:
        
        """can only be used for quadratic cost function"""

        # Transform state and input traj from list to ndarray
        state_traj = np.array(self.state_traj)
        input_traj = np.array(self.input_traj)

        total_cost_arr = np.zeros(len(input_traj))
        
        # Terminal cost
        x_final = state_traj[-1]
        x_err_final = x_final - target_state
        cost_terminal = 0.5 * (x_err_final.T @ Qf @ x_err_final)
        total_cost_arr[-1] = cost_terminal

        # Stage cost, backpropagation
        for k in range(len(input_traj)-2, -1, -1):
            x_err = state_traj[k] - target_state
            u = input_traj[k]

            u_cost = R * (u**2)

            cost_stage = 0.5 * (x_err.T @ Q @ x_err + u_cost)
            total_cost_arr[k] = total_cost_arr[k+1] +  cost_stage
        
        # Update the cost2go_arr
        self.cost2go_arr = total_cost_arr

        return total_cost_arr







class Visualizer:
    def __init__(
        self, 
        simulator: Simulator
    ) -> None:

        self.simulator = simulator
        self.dynamics = simulator.dynamics
        self.controller = simulator.controller
        self.env = simulator.env

        self.state_traj, self.input_traj = self.simulator.get_trajectories()

        self.position = self.state_traj[:, 0]
        self.velocity = self.state_traj[:, 1]
        self.acceleration = self.input_traj
        
        self.dt = simulator.dt
        self.t_eval = simulator.t_eval

        self.delta_index_pred_display = 2

        # Define setting of plots and animation
        self.color = "blue"
        self.color_list = ['red', 'green', 'yellow', 'orange']

        self.shadow_space_wide = 0.2

        self.car_length= 0.2
        self.figsize = (8, 4)
        self.refresh_rate = 30

    def display_plots(self, title = None) -> None:

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))

        # Plot predicted positions
        if self.simulator.controller_type in ['MPC', 'RMPC'] and len(self.simulator.state_pred_traj)>0:
            for i in range(len(self.simulator.state_pred_traj)):
                if i%self.delta_index_pred_display == 0:
                    state_pred_traj_curr = self.simulator.state_pred_traj[i]
                    t_eval = np.linspace(self.t_eval[i], self.t_eval[i] + self.simulator.dt * (len(state_pred_traj_curr) - 1), len(state_pred_traj_curr))
                    ax1.plot(t_eval, state_pred_traj_curr[:, 0], label="Predicted Position", linestyle="--", color="orange")
        
        # Plot reference if have
        if hasattr(self.simulator.controller, 'traj_ref'):
            ax1.plot(self.t_eval, self.simulator.controller.traj_ref[:, 0], color='gray', marker='.', linestyle='-', label='Reference Position')

        # Plot p over time t
        ax1.plot(self.t_eval, self.position, label="Position p(t)", color="blue")

        # Plot shadowed zone for state bounds
        if self.env.state_lbs is not None:
            ymin, _ = ax1.get_ylim()
            lower_edge = min(self.env.state_lbs[0] - self.shadow_space_wide, ymin)
            upper_edge = self.env.state_lbs[0]
            ax1.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'pos_lower_bound')
        if self.env.state_ubs is not None:
            _, ymax = ax1.get_ylim()
            lower_edge = self.env.state_ubs[0]
            upper_edge = max(self.env.state_ubs[0] + self.shadow_space_wide, ymax)
            ax1.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'pos_upper_bound')

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())

        # Plot predicted velocities
        if self.simulator.controller_type in ['MPC', 'RMPC'] and len(self.simulator.state_pred_traj)>0:
            for i in range(len(self.simulator.state_pred_traj)):
                if i%self.delta_index_pred_display == 0:
                    state_pred_traj_curr = self.simulator.state_pred_traj[i]
                    t_eval = np.linspace(self.t_eval[i], self.t_eval[i] + self.simulator.dt * (len(state_pred_traj_curr) - 1), len(state_pred_traj_curr))
                    ax2.plot(t_eval, state_pred_traj_curr[:, 1], label="Predicted Velocity", linestyle="--", color="orange")
        
        # Plot reference if have
        if hasattr(self.simulator.controller, 'traj_ref'):
            ax2.plot(self.t_eval, self.simulator.controller.traj_ref[:, 1], color='gray', marker='.', linestyle='-', label='Reference Velocity')

        # Plot v over time t
        ax2.plot(self.t_eval, self.velocity, label="Velocity v(t)", color="green")

        # Plot shadowed zone for state bounds
        if self.env.state_lbs is not None:
            ymin, _ = ax2.get_ylim()
            lower_edge = min(self.env.state_lbs[1] - self.shadow_space_wide, ymin)
            upper_edge = self.env.state_lbs[1]
            ax2.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'vel_lower_bound')
        if self.env.state_ubs is not None:
            _, ymax = ax2.get_ylim()
            lower_edge = self.env.state_ubs[1]
            upper_edge = max(self.env.state_ubs[1] + self.shadow_space_wide, ymax)
            ax2.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'vel_upper_bound')

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys())

        '''
        # Not updated yet, still continuous signal instead of ZOH signal
        # Plot predicted accelerations
        if self.simulator.controller_type in ['MPC', 'RMPC'] and len(self.simulator.state_pred_traj)>0:
            for i in range(len(self.simulator.state_pred_traj)):
                if i%self.delta_index_pred_display == 0:
                    input_pred_traj_curr = self.simulator.input_pred_traj[i]
                    t_eval = np.linspace(self.t_eval[i], self.t_eval[i] + self.simulator.dt * (len(input_pred_traj_curr) - 1), len(input_pred_traj_curr))
                    ax3.plot(t_eval, input_pred_traj_curr, label="Predicted Input", linestyle="--", color="orange")
        '''

        # Plot a over time t
        #ax3.plot(self.t_eval[:-1], self.acceleration, label="Input u(t)", color="red")
        ax3.step(self.t_eval, np.append(self.acceleration, self.acceleration[-1]), where='post', label="Input u(t)", color='red')
        #ax3.plot(self.t_eval[:-1], self.acceleration, 'o', color='red')
        if self.simulator.controller_type == 'RMPC':
            nominal_acceleration = self.simulator.nominal_input_traj
            ax3.step(self.t_eval, np.append(nominal_acceleration, nominal_acceleration[-1]), where='post', label="Nominal Input u(t)", color='purple')

        # Plot shadowed zone for input bounds
        if self.env.input_lbs is not None:
            ymin, _ = ax3.get_ylim()
            lower_edge = min(self.env.input_lbs-self.shadow_space_wide, ymin)
            upper_edge = self.env.input_lbs
            ax3.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'input_lower_bound')
        if self.env.input_ubs is not None:
            _, ymax = ax3.get_ylim()
            lower_edge = self.env.input_ubs
            upper_edge = max(self.env.input_ubs+self.shadow_space_wide, ymax)
            ax3.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'input_upper_bound')
            
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Input (m/s^2)")

        handles, labels = ax3.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax3.legend(by_label.values(), by_label.keys())
        
        if title is not None:
            fig.suptitle(title, fontsize=20, fontweight='bold', y=1.05)
            
        plt.tight_layout()
        plt.show()

    def display_contrast_plots(self, *simulators: Simulator, title=None, if_gray:bool=False) -> None:

        color_index = 0

        if not simulators:
            raise ValueError("No simulator references provided.")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))

        # Plot current object's trajectories
        ax1.plot(self.t_eval, self.position, label=f"{self.simulator.controller_name}", color=self.color)

        # Plot reference if have
        if hasattr(self.simulator.controller, 'traj_ref'):
            ax1.plot(self.t_eval, self.simulator.controller.traj_ref[:, 0], color='gray', marker='.', linestyle='-', label='Reference Trajectory')

        ax2.plot(self.t_eval, self.velocity, label=f"{self.simulator.controller_name}", color=self.color)

        # Plot reference if have
        if hasattr(self.simulator.controller, 'traj_ref'):
            ax2.plot(self.t_eval, self.simulator.controller.traj_ref[:, 1], color='gray', marker='.', linestyle='-', label='Reference Trajectory')

        #ax3.plot(self.t_eval[:-1], self.acceleration, label=f"{self.simulator.controller_name}", color=self.color)
        ax3.step(self.t_eval, np.append(self.acceleration, self.acceleration[-1]), where='post', label=f"{self.simulator.controller_name}", color=self.color)

        # Plot the reference and evaluated trajectories for each simulator
        for simulator_ref in simulators:
            if simulator_ref.state_traj is None or len(simulator_ref.state_traj) == 0:
                raise ValueError(f"Failed to get trajectory from simulator {simulator_ref.controller_name}. State trajectory list is void; please run 'run_simulation' first.")

            # Get reference trajectories from simulator_ref
            state_traj_ref, input_traj_ref = simulator_ref.get_trajectories()

            # Extract reference position, velocity, and acceleration
            position_ref = state_traj_ref[:, 0]
            velocity_ref = state_traj_ref[:, 1]
            acceleration_ref = input_traj_ref

            if if_gray:
                # Plot position over time
                ax1.plot(simulator_ref.t_eval, position_ref, linestyle="--", label=f"{simulator_ref.controller_name}", color='gray')
                # Plot velocity over time
                ax2.plot(simulator_ref.t_eval, velocity_ref, linestyle="--", label=f"{simulator_ref.controller_name}", color='gray')
                # Plot acceleration over time
                ax3.step(simulator_ref.t_eval, np.append(acceleration_ref, acceleration_ref[-1]), where='post', linestyle="--", label=f"{simulator_ref.controller_name}", color='gray')
            else:
                # Plot position over time
                ax1.plot(simulator_ref.t_eval, position_ref, linestyle="--", label=f"{simulator_ref.controller_name}", color=self.color_list[color_index])
                # Plot velocity over time
                ax2.plot(simulator_ref.t_eval, velocity_ref, linestyle="--", label=f"{simulator_ref.controller_name}", color=self.color_list[color_index])
                # Plot acceleration over time
                ax3.step(simulator_ref.t_eval, np.append(acceleration_ref, acceleration_ref[-1]), where='post', linestyle="--", label=f"{simulator_ref.controller_name}", color=self.color_list[color_index])

                color_index += 1
            
        # Plot shadowed zone for state bounds
        if self.env.state_lbs is not None:
            ymin, _ = ax1.get_ylim()
            lower_edge = min(self.env.state_lbs[0] - self.shadow_space_wide, ymin)
            upper_edge = self.env.state_lbs[0]
            ax1.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'pos_lower_bound')
        if self.env.state_ubs is not None:
            _, ymax = ax1.get_ylim()
            lower_edge = self.env.state_ubs[0]
            upper_edge = max(self.env.state_ubs[0] + self.shadow_space_wide, ymax)
            ax1.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'pos_upper_bound')

        if self.env.state_lbs is not None:
            ymin, _ = ax2.get_ylim()
            lower_edge = min(self.env.state_lbs[1] - self.shadow_space_wide, ymin)
            upper_edge = self.env.state_lbs[1]
            ax2.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'vel_lower_bound')
        if self.env.state_ubs is not None:
            _, ymax = ax2.get_ylim()
            lower_edge = self.env.state_ubs[1]
            upper_edge = max(self.env.state_ubs[1] + self.shadow_space_wide, ymax)
            ax2.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'vel_upper_bound')
            
        if self.env.input_lbs is not None:
            ymin, _ = ax3.get_ylim()
            lower_edge = min(self.env.input_lbs-self.shadow_space_wide, ymin)
            upper_edge = self.env.input_lbs
            ax3.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'input_lower_bound')
        if self.env.input_ubs is not None:
            _, ymax = ax3.get_ylim()
            lower_edge = self.env.input_ubs
            upper_edge = max(self.env.input_ubs+self.shadow_space_wide, ymax)
            ax3.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'input_upper_bound')


        # Set labels and legends
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.legend()

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.legend()

        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Input (m/s^2)")
        ax3.legend()

        if title is not None:
            fig.suptitle(title, fontsize=20, fontweight='bold', y=1.05)

        plt.tight_layout()
        plt.show()

    def display_contrast_cost2go(self, *simulators: Simulator) -> None:

        color_index = 0

        if not simulators:
            raise ValueError("No simulator references provided.")

        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        gs.update(hspace=0.4, wspace=0.3)

        # cost vs time
        ax0 = fig.add_subplot(gs[0, 0]) 
        ax0.set_title("Total Cost w.r.t. simulation time")

        # cost vs iteration
        ax1 = fig.add_subplot(gs[1, :])       
        ax1.set_title("Total Cost w.r.t. iLQR Iteration")

        ax = [ax0, ax1] 

        # Plot the reference and evaluated trajectories for each simulator
        for simulator_ref in simulators:
            if not np.all(simulator_ref.cost2go_arr):
                raise ValueError(f"Failed to get trajectory from simulator {simulator_ref.controller_name}. State trajectory list is void; please run 'run_simulation' first.")

            # Plot cost over time
            ax[0].plot(simulator_ref.t_eval, np.append(simulator_ref.cost2go_arr, simulator_ref.cost2go_arr[-1]), linestyle="--", label=f"{simulator_ref.controller_name}", color=self.color_list[color_index])

            color_index += 1
        
            # Annotate total cost at initial time step
            initial_cost = simulator_ref.cost2go_arr[0]
            ax[0].annotate(
                f"Total cost: {initial_cost:.2f}",
                xy=(0, initial_cost),
                xytext=(10, initial_cost + 0.05 * initial_cost),
                arrowprops=dict(arrowstyle="->", lw=1.5),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
            )

        # Plot cost over time
        ax[0].plot(self.t_eval, np.append(self.simulator.cost2go_arr, self.simulator.cost2go_arr[-1]), label=f"{self.simulator.controller_name}", color=self.color_list[color_index])

        color_index += 1
    
        # Annotate total cost at initial time step
        initial_cost = self.simulator.cost2go_arr[0]
        ax[0].annotate(
            f"Total cost: {initial_cost:.2f}",
            xy=(0, initial_cost),
            xytext=(10, initial_cost + 0.05 * initial_cost),
            arrowprops=dict(arrowstyle="->", lw=1.5),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
        )

        # Set labels and legends
        ax[0].set_title("Total Cost w.r.t. simulation time")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Cost-to-go")
        ax[0].legend()

        # Show iLQR total cost w.r.t iterations
        if self.simulator.controller_type == 'iLQR': 

            ilqr_total_cost_list = self.simulator.controller.total_cost_list
            ilqr_total_cost_list[0] = simulator_ref.cost2go_arr[0]

            ax[1].plot(ilqr_total_cost_list, marker='o', label="iLQR")

            for simulator_ref in simulators:
                ax[1].axhline(y=simulator_ref.cost2go_arr[0], linestyle='--', color='red', label="LQR")

            ax[1].set_title("Total Cost w.r.t. iLQR Iteration")
            ax[1].set_xlabel("Iteration")
            ax[1].set_ylabel("Total Cost")
            ax[1].legend()
            ax[1].grid(True)

        plt.show()
        
    def display_phase_portrait(self):

        """
        Display RMPC results in the (x1, x2) plane, showing nominal trajectory,
        true trajectory, and invariant tube cross-sections.
        """
        
        if self.controller.type != 'RMPC':
            raise ValueError("This visualization is only supported for Tube-based RMPC controllers.")

        fig, ax = plt.subplots(figsize=(6, 6))

        # Extract true state trajectory
        x_true = self.state_traj[:, 0]
        v_true = self.state_traj[:, 1]
        
        # Extract nominal trajectory if available
        x_nom = []
        v_nom = []
        for i in range(len(self.simulator.state_pred_traj)):
            state_pred_traj_curr = self.simulator.state_pred_traj[i]
            x_nom.append(state_pred_traj_curr[0, 0])
            v_nom.append(state_pred_traj_curr[0, 1])

        # Plot nominal and true trajectories
        ax.plot(x_nom, v_nom, linestyle='--', color='black', marker='x', label='Nominal Trajectory')
        ax.plot(x_true, v_true, linestyle='-', color='blue', marker='*', label='Real Trajectory')

        # Plot red tube polygons (Ω translated to nominal)
        Omega = self.controller.Omega_tube.V
        assert Omega.shape[1] == 2, "Only 2D invariant sets are supported."
        # Compute axis-aligned bounding box
        min_bounds = np.min(self.controller.Omega_tube.V, axis=0)
        max_bounds = np.max(self.controller.Omega_tube.V, axis=0)

        for i in range(len(x_nom)):

            # Show polytopes
            center = np.array([x_nom[i], v_nom[i]])
            tube_vertices = Omega + center  # Translate tube
            if i == 0:
                patch = Polygon(tube_vertices, closed=True, edgecolor='red', facecolor='red', alpha=0.3, label='Robust Invariant Set Ω')
            else:
                patch = Polygon(tube_vertices, closed=True, edgecolor='red', facecolor='red', alpha=0.3)
            ax.add_patch(patch)

            # Show bounding box
            #bounding_box = np.array([
            #    [min_bounds[0]+x_nom[i], min_bounds[1]+v_nom[i]],
            #    [min_bounds[0]+x_nom[i], max_bounds[1]+v_nom[i]],
            #    [max_bounds[0]+x_nom[i], max_bounds[1]+v_nom[i]],
            #    [max_bounds[0]+x_nom[i], min_bounds[1]+v_nom[i]],
            #    [min_bounds[0]+x_nom[i], min_bounds[1]+v_nom[i]]
            #])  
            # Translate bounding box
            #ax.plot(bounding_box[:, 0], bounding_box[:, 1], 'r--', linewidth=2)
        
        # Show initial state and target state
        ax.plot(self.env.initial_position, self.env.initial_velocity, marker='o', color='darkorange', markersize=12, markeredgewidth=3, label='Initial State')
        ax.plot(self.env.target_position, self.env.target_velocity, marker='x', color='green', markersize=15, markeredgewidth=3, label='Target State')

        ax.set_xlabel(r"Position $p$")
        ax.set_ylabel(r"Velocity $v$")
        #ax.set_xlim(self.simulator.env.pos_lbs, self.simulator.env.pos_ubs)
        #ax.set_ylim(self.simulator.env.vel_lbs, self.simulator.env.vel_ubs)
        ax.set_title("Trajectory of RMPC in Phase Portrait with Tube Ω")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def display_animation(self) -> HTML:
        
        # Instantiate the plotting
        fig, ax1 = plt.subplots(1, 1, figsize=self.figsize)
        
        # Define size of plotting
        p_max = 1.0 #max(self.position)
        p_min = -1.0 #min(self.position)
        start_extension = p_min - 0.3
        end_extension = p_max + 0.3

        p_disp_vals = np.linspace(start_extension, end_extension, 200) # generate grid mesh on p

        h_disp_vals = [float(self.env.h(p).full().flatten()[0]) for p in p_disp_vals]
        h_max = 1.0 #max(h_disp_vals)
        h_min = -1.0 #min(h_disp_vals)

        ax1.set_xlim(start_extension-0.5, end_extension+0.5)
        ax1.set_ylim(h_min-0.3, h_max+0.3)

        # Draw mountain profile curve h(p)
        ax1.plot(p_disp_vals, h_disp_vals, label="Mountain profile h(p)", color="green")
        ax1.set_xlabel("Position p")
        ax1.set_ylabel("Height h")
        ax1.legend()

        # Mark the intial state and the target state in the plotting
        initial_h = float(self.env.h(self.env.initial_position).full().flatten()[0])
        target_h = float(self.env.h(self.env.target_position).full().flatten()[0])
        ax1.scatter([self.env.initial_position], [initial_h], color="blue", label="Start")
        #ax1.scatter([self.env.target_position], [target_h], color="orange", label="Target position")
        ax1.plot(self.env.target_position, target_h, marker='x', color='red', markersize=10, markeredgewidth=3, label='Target')

        if self.controller.type == 'LQR' and not np.allclose(self.controller.state_lin, self.controller.target_state):
            lin_position = self.controller.state_lin[0]
            lin_h = float(self.env.h(lin_position).full().flatten()[0])
            ax1.plot(lin_position, lin_h, marker='v', color='orange', markersize=7, markeredgewidth=3, label='Linearization Point')

        ax1.legend()


        # Setting simplyfied car model as rectangle, and update the plotting to display the animation
        car_height = self.car_length / 2
        car = Rectangle((0, 0), self.car_length, car_height, color="black")
        ax1.add_patch(car)

        def update(frame):
            # Get current position and attitude of car
            current_position = self.position[frame]
            current_theta = float(self.env.theta(current_position).full().flatten()[0])

            # Update position and attitude of car
            car.set_xy((current_position - self.car_length / 2, float(self.env.h(current_position - self.car_length / 2).full().flatten()[0])))
            car.angle = np.degrees(current_theta)  # rad to deg

        # Instantiate animation
        anim = FuncAnimation(fig, update, frames=len(self.t_eval), interval=1000 / self.refresh_rate, repeat=False)

        plt.close(fig)

        return HTML(anim.to_jshtml())
    
    def display_contrast_animation(self, *simulators) -> HTML:

        # Instantiate the plotting
        num_plots = len(simulators) + 1
        fig, axes = plt.subplots(num_plots, 1, figsize=(self.figsize[0], self.figsize[1] * num_plots), sharex=True)

        # Define size of plotting
        p_max = 1.0 #max(max(sim.get_trajectories()[0][:, 0]) for sim in simulators)
        p_min = -1.0 #min(min(sim.get_trajectories()[0][:, 0]) for sim in simulators)
        start_extension = p_min - 0.3
        end_extension = p_max + 0.3

        p_disp_vals = np.linspace(start_extension, end_extension, 200)  # generate grid mesh on p

        h_disp_vals = [float(self.env.h(p).full().flatten()[0]) for p in p_disp_vals]
        h_max = 1.0 #max(h_disp_vals)
        h_min = -1.0 #min(h_disp_vals)
        axes[0].set_xlim(start_extension - 0.5, end_extension + 0.5)
        axes[0].set_ylim(h_min - 0.3, h_max + 0.3)
        axes[0].plot(p_disp_vals, h_disp_vals, label="Mountain profile h(p)", color="green")
        axes[0].set_xlabel("Position p")
        axes[0].set_ylabel("Height h")

        for ax, sim in zip(axes[1:], simulators):

            h_disp_vals = [float(sim.env.h(p).full().flatten()[0]) for p in p_disp_vals]
            ax.set_xlim(start_extension - 0.5, end_extension + 0.5)
            ax.set_ylim(h_min - 0.3, h_max + 0.3)
            ax.plot(p_disp_vals, h_disp_vals, label="Mountain profile h(p)", color="green")
            ax.set_xlabel("Position p")
            ax.set_ylabel("Height h")


        # Mark the initial state and the target state in the plotting
        initial_h = float(self.env.h(self.env.initial_position).full().flatten()[0])
        target_h = float(self.env.h(self.env.target_position).full().flatten()[0])
        axes[0].scatter([self.env.initial_position], [initial_h], color="blue", label="Start")
        #axes[0].scatter([self.env.target_position], [target_h], color="orange", label="Target position")
        axes[0].plot(self.env.target_position, target_h, marker='x', color='red', markersize=10, markeredgewidth=3, label='Target')
        axes[0].legend()

        for ax, sim in zip(axes[1:], simulators):

            initial_h = float(sim.env.h(self.env.initial_position).full().flatten()[0])
            target_h = float(sim.env.h(self.env.target_position).full().flatten()[0])

            ax.scatter([sim.env.initial_position], [initial_h], color="blue", label="Start")

            #ax.scatter([sim.env.target_position], [target_h], color="orange", label="Target position")
            ax.plot(sim.env.target_position, target_h, marker='x', color='red', markersize=10, markeredgewidth=3, label='Target')


            if sim.controller.type == 'LQR' and not np.allclose(sim.controller.state_lin, sim.controller.target_state):
                lin_position = sim.controller.state_lin[0]
                lin_h = float(sim.env.h(lin_position).full().flatten()[0])
                ax.plot(lin_position, lin_h, marker='v', color='orange', markersize=7, markeredgewidth=3, label='Linearization Point of LQR')

            ax.legend()


        # Create car objects for each simulator
        car_objects = {}
        colors = self.color_list[:len(simulators)]
        car_height = self.car_length / 2

        car_self = Rectangle((0, 0), self.car_length, car_height, color=self.color)
        axes[0].add_patch(car_self)
        axes[0].set_title(f"{self.simulator.controller_name}")

        for ax, sim, color in zip(axes[1:], simulators, colors):
            car = Rectangle((0, 0), self.car_length, car_height, color=color)
            ax.add_patch(car)
            ax.set_title(f"{sim.controller_name}")
            car_objects[sim] = car


        def update(frame):
            # Update car for self
            current_position = self.position[frame]
            current_theta = float(self.env.theta(current_position).full().flatten()[0])
            car_self.set_xy((current_position - self.car_length / 2, float(self.env.h(current_position - self.car_length / 2).full().flatten()[0])))
            car_self.angle = np.degrees(current_theta)  # rad to deg

            for sim, car in car_objects.items():
                frame_ref = int(frame*self.dt/sim.dt)  # Adjust frame index for each simulator
                current_position = sim.get_trajectories()[0][:, 0][frame_ref]
                current_theta = float(sim.env.theta(current_position).full().flatten()[0])
                car.set_xy((current_position - self.car_length / 2, float(sim.env.h(current_position - self.car_length / 2).full().flatten()[0])))
                car.angle = np.degrees(current_theta)  # rad to deg

        # Instantiate animation
        anim = FuncAnimation(fig, update, frames=len(self.t_eval), interval=1000 / self.refresh_rate, repeat=False)

        plt.close(fig)

        return HTML(anim.to_jshtml())
    
    def display_contrast_animation_same(self, *simulators, if_gray:bool = False) -> HTML:
        import matplotlib.patches as mpatches

        custom_handles = []

        # Setup figure
        fig, ax = plt.subplots(1, 1, figsize=(self.figsize[0], self.figsize[1]))

        # Plot mountain profile
        p_max, p_min = 1.0, -1.0
        start_extension, end_extension = p_min - 0.3, p_max + 0.3
        p_disp_vals = np.linspace(start_extension, end_extension, 200)
        h_disp_vals = [float(self.env.h(p).full().flatten()[0]) for p in p_disp_vals]

        ax.set_xlim(start_extension - 0.5, end_extension + 0.5)
        ax.set_ylim(-1.3, 1.3)
        profile_plot, = ax.plot(p_disp_vals, h_disp_vals, label="Mountain profile h(p)", color="green")
        custom_handles.append(profile_plot)

        ax.set_xlabel("Position p")
        ax.set_ylabel("Height h")

        # Start & Target markers
        initial_h = float(self.env.h(self.env.initial_position).full().flatten()[0])
        target_h = float(self.env.h(self.env.target_position).full().flatten()[0])
        start_scatter = ax.scatter([self.env.initial_position], [initial_h], color="blue", label="Start")
        custom_handles.append(start_scatter)
        target_cross = ax.plot(self.env.target_position, target_h, marker='x', color='red', markersize=10, markeredgewidth=3, label='Target')[0]
        custom_handles.append(target_cross)

        for sim in simulators:

            if sim.controller.type == 'LQR' and not np.allclose(sim.controller.state_lin, sim.controller.target_state):
                lin_position = sim.controller.state_lin[0]
                lin_h = float(sim.env.h(lin_position).full().flatten()[0])
                lin_point = ax.plot(lin_position, lin_h, marker='v', color='orange', markersize=7, markeredgewidth=3, label='Linearization Point of LQR')[0]
                custom_handles.append(lin_point)

        # Car setup
        car_objects = {}
        colors = [self.color] + self.color_list[:len(simulators)]
        car_height = self.car_length / 2

        # Self car
        car_self = Rectangle((0, 0), self.car_length, car_height,
                            edgecolor=self.color, facecolor='none', linewidth=2)
        ax.add_patch(car_self)
        car_objects[self] = car_self

        # Simulators' cars
        for sim, color in zip(simulators, colors[1:]):
            if if_gray:
                car = Rectangle((0, 0), self.car_length, car_height, edgecolor='gray', facecolor='none', linewidth=2)
            else:
                car = Rectangle((0, 0), self.car_length, car_height, edgecolor=color, facecolor='none', linewidth=2)
            ax.add_patch(car)
            car_objects[sim] = car

        # Title using correct controller names
        controller_names = " vs. ".join(
            [self.simulator.controller_name] + [sim.controller_name for sim in simulators]
        )
        ax.set_title(controller_names)

        # Legend

        car_legend_handles = [
            mpatches.Patch(edgecolor=self.color, facecolor='none', linewidth=2, label=self.simulator.controller_name)
        ]
        for sim, color in zip(simulators, colors[1:]):
            if if_gray:
                car_legend_handles.append(mpatches.Patch(edgecolor='gray', facecolor='none', linewidth=2, label=sim.controller_name))
            else:
                car_legend_handles.append(mpatches.Patch(edgecolor=color, facecolor='none', linewidth=2, label=sim.controller_name))

        ax.legend(handles=custom_handles + car_legend_handles, loc='best')

        # Animation update function
        def update(frame):
            # Self
            current_position = self.position[frame]
            current_theta = float(self.env.theta(current_position).full().flatten()[0])
            y_base = float(self.env.h(current_position - self.car_length / 2).full().flatten()[0])
            car_self.set_xy((current_position - self.car_length / 2, y_base))
            car_self.angle = np.degrees(current_theta)

            # Simulators
            for sim, car in car_objects.items():
                if sim is self:
                    continue
                frame_ref = int(frame*self.dt/sim.dt)  # Adjust frame index for each simulator
                current_position = sim.get_trajectories()[0][:, 0][frame_ref]
                current_theta = float(sim.env.theta(current_position).full().flatten()[0])
                y_base = float(sim.env.h(current_position - self.car_length / 2).full().flatten()[0])
                car.set_xy((current_position - self.car_length / 2, y_base))
                car.angle = np.degrees(current_theta)

        # Animate
        anim = FuncAnimation(fig, update, frames=len(self.t_eval),
                            interval=1000 / self.refresh_rate, repeat=False)
        
        plt.close(fig)
        
        return HTML(anim.to_jshtml())