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
from matplotlib.patches import Polygon
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






def draw_flag(ax, p_flag, h_func, normal_vec, 
              flag_height=0.35, flag_width=0.20,
              pole_color="black", flag_color="red"):
    """
    Draw a small flag on the mountain profile.

    Arguments:
        ax: Axes object to draw on.
        p_flag: float, horizontal position of the flag.
        h_func: function(p)->height, your env.h.
        normal_vec: array-like (2,), direction vector indicating flag’s vertical direction.
        flag_height: height of flag pole.
        flag_width: width of the triangular flag.
    """

    # Normalize the normal vector
    n = np.array(normal_vec, dtype=float)
    n = n / np.linalg.norm(n)

    # Orthogonal direction for flag triangle
    t = np.array([n[1], -n[0]])  # 90 degrees rotated

    # Base point of the flag on the curve
    h_flag = float(h_func(p_flag).full().flatten()[0])
    base = np.array([p_flag, h_flag])

    # Pole top = base + n * flag_height
    pole_bottom = base + n * flag_height
    pole_top = pole_bottom + n * 0.08

    # Triangular flag vertices
    v1 = pole_bottom # left down
    v2 = pole_bottom + n * 0.08 # left up
    v3 = pole_bottom + t * flag_width # right

    # Draw pole
    ax.plot([base[0], pole_top[0]],
            [base[1], pole_top[1]],
            color=pole_color, linewidth=2)

    # Draw flag triangle
    tri = Polygon([v1, v2, v3], color=flag_color, alpha=0.85)
    ax.add_patch(tri)

def add_car(ax, car_length, wheel_radius, color="steelblue", filled=True):

    # Create car shape
    # # Shape 1
    # car_shape = np.array([
    #     [-3*car_length/4+car_length/8,  -wheel_radius],
    #     [-3*car_length/4+car_length/8,  car_length/4-wheel_radius],
    #     [-car_length/2+car_length/8,    car_length/4-wheel_radius],
    #     [-car_length/4+car_length/8,    car_length/2-wheel_radius],
    #     [car_length/4-car_length/8,     car_length/2-wheel_radius],
    #     [car_length/2-car_length/8,     car_length/4-wheel_radius],
    #     [3*car_length/4-car_length/8,   car_length/4-wheel_radius],
    #     [4*car_length/5-car_length/8,   -wheel_radius],
    #     [-3*car_length/4+car_length/8,  -wheel_radius],
    # ])
    # Shape 2: Jeep
    car_shape = np.array([
        [-3*car_length/4+car_length/8,    -wheel_radius],
        [-2*car_length/3+car_length/8,    car_length/2-wheel_radius],
        [car_length/4-car_length/8,       car_length/2-wheel_radius],
        [car_length/2-car_length/8,       car_length/4-wheel_radius],
        [3*car_length/4-car_length/8,     car_length/4-wheel_radius],
        [4*car_length/5-car_length/8,     -wheel_radius],
        [-3*car_length/4+car_length/8,    -wheel_radius],
    ])
    # Filled or outline?
    if filled:
        car_body = Polygon(
            transform_shape(car_shape, [0,0], 0),
            closed=True, facecolor=color, edgecolor=color, linewidth=2
        )
    else:
        car_body = Polygon(
            transform_shape(car_shape, [0,0], 0),
            closed=True, facecolor='none', edgecolor=color, linewidth=2
        )
    ax.add_patch(car_body)
    # Wheels
    wheel_left = plt.Circle((0, 0), wheel_radius, color="black")
    wheel_right = plt.Circle((0, 0), wheel_radius, color="black")
    ax.add_patch(wheel_left)
    ax.add_patch(wheel_right)
    
    return car_shape, car_body, wheel_left, wheel_right
    
def transform_shape(shape, center, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return shape @ R.T + center

def solve_wheel_center(wheel_center_x, r, env):
    # variable
    x_star = ca.SX.sym('x_star')

    # expression F(x*) = x* + r*sin(theta(x*)) - x
    theta = env.theta(x_star)          # CasADi function
    F = x_star - r * ca.sin(theta) - wheel_center_x

    # solver
    Ffun = ca.Function("Ffun", [x_star], [F])
    solver = ca.rootfinder("solver", "newton", Ffun)
    
    # initial guess
    x0 = wheel_center_x

    sol = solver(x0)
    
    height = env.h(sol) + r * ca.cos(env.theta(sol))

    return (float(wheel_center_x), float(height))

def animate(env, p, car_length, wheel_radius, axle_offset,
            car_shape, car_body, wheel_left, wheel_right):
    
    theta = float(env.theta(p).full().flatten()[0])

    # Wheel
    w1_x = p - (car_length/2-wheel_radius) * np.cos(theta) - wheel_radius * np.sin(theta)
    w2_x = p + (car_length/2-wheel_radius) * np.cos(theta) - wheel_radius * np.sin(theta)
    (w1_x, w1_y) = solve_wheel_center(w1_x, wheel_radius, env)
    (w2_x, w2_y) = solve_wheel_center(w2_x, wheel_radius, env)
    # Rectangle
    theta_real = np.arctan2((w2_y - w1_y), (w2_x - w1_x))
    body_x_center = (w1_x + w2_x) / 2 - axle_offset * np.sin(theta_real)
    body_y_center = (w1_y + w2_y) / 2 + axle_offset * np.cos(theta_real)
    # Update wheels
    wheel_left.center = (w1_x, w1_y)
    wheel_right.center = (w2_x, w2_y)
    # Update body
    car_center = np.array([body_x_center, body_y_center])
    shape_world = transform_shape(car_shape, car_center, theta_real)
    car_body.set_xy(shape_world)

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
        
        self.figsize = (6, 6) if self.env.case == 4 else (6, 3)
        self.shadow_space_wide = 0.2
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

    def display_animation(self, if_save=False) -> HTML:
        
        # Instantiate the plotting
        fig, ax1 = plt.subplots(1, 1, figsize=self.figsize)
        
        # Define size of plotting
        p_max = 1.0 #max(self.position)
        p_min = -1.0 #min(self.position)
        start_extension = p_min - 0.3
        end_extension = p_max + 0.3

        p_disp_vals = np.linspace(start_extension, end_extension, 200) # generate grid mesh on p
        h_disp_vals = [float(self.env.h(p).full().flatten()[0]) for p in p_disp_vals]
        ax1.set_xlim(start_extension, end_extension)
        ax1.set_ylim(-1.3, 2.0 if self.simulator.env.case == 4 else 1.3)

        # Draw mountain profile curve h(p)
        ax1.plot(p_disp_vals, h_disp_vals, label="Mountain profile h(p)", color="black")
        ax1.set_xlabel("Position p")
        ax1.set_ylabel("Height h")
        ax1.fill_between(
            p_disp_vals,
            h_disp_vals,
            -1.3,
            color="gray",
            alpha=0.3
        )

        # Draw a mountain-aligned flag at the target
        p_flag = self.env.target_position
        theta_flag = float(self.env.theta(p_flag).full().flatten()[0])
        normal_vec = np.array([np.sin(theta_flag), np.cos(theta_flag)])
        draw_flag(ax1, p_flag=p_flag, h_func=self.env.h,
                  normal_vec=((0, 1) if self.simulator.env.case == 3 else normal_vec))

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
        ax1.legend(loc='lower right')

        # Car parameters
        car_length= 0.2
        wheel_radius = 0.035
        axle_offset = 1.5 * wheel_radius   # height from wheel center to car bottom

        # Build car body + wheels
        self.car_shape, self.car_body, self.wheel_left, self.wheel_right = add_car(ax1, car_length, wheel_radius)
        
        # Animation update function
        def update(frame):
            # Ideal car center position
            p = self.position[frame]
            # Animate
            animate(self.env, p, car_length, wheel_radius, axle_offset, self.car_shape, self.car_body, self.wheel_left, self.wheel_right)


        # Instantiate animation
        anim = FuncAnimation(fig, update, frames=len(self.t_eval), interval=1000 / self.refresh_rate, repeat=False)
        
        # Save animation if required
        if if_save:
            anim.save("animation.mp4", writer="ffmpeg", fps=self.refresh_rate)

        plt.close(fig)

        return HTML(anim.to_jshtml())
    
    def display_contrast_animation(self, *simulators, if_gray:bool = False, if_save=False) -> HTML:
        import matplotlib.patches as mpatches

        custom_handles = []

        # Setup figure
        fig, ax = plt.subplots(1, 1, figsize=(self.figsize[0], self.figsize[1]))

        # Plot mountain profile
        p_max, p_min = 1.0, -1.0
        start_extension, end_extension = p_min - 0.3, p_max + 0.3
        p_disp_vals = np.linspace(start_extension, end_extension, 200)
        h_disp_vals = [float(self.env.h(p).full().flatten()[0]) for p in p_disp_vals]

        ax.set_xlim(start_extension, end_extension)
        ax.set_ylim(-1.3, 2.0 if self.simulator.env.case == 4 else 1.3)
        profile_plot, = ax.plot(p_disp_vals, h_disp_vals, label="Mountain profile h(p)", color="black")
        ax.fill_between(
            p_disp_vals,
            h_disp_vals,
            -1.3,
            color="gray",
            alpha=0.3
        )
        custom_handles.append(profile_plot)
        ax.set_xlabel("Position p")
        ax.set_ylabel("Height h")

        # Draw a mountain-aligned flag at the target
        p_flag = self.env.target_position
        theta_flag = float(self.env.theta(p_flag).full().flatten()[0])
        normal_vec = np.array([np.sin(theta_flag), np.cos(theta_flag)])
        draw_flag(ax,
                p_flag=p_flag,
                h_func=self.env.h,
                normal_vec=((0, 1) if self.simulator.env.case == 3 else normal_vec))

        # Start & Target markers
        initial_h = float(self.env.h(self.env.initial_position).full().flatten()[0])
        target_h = float(self.env.h(self.env.target_position).full().flatten()[0])
        start_scatter = ax.scatter([self.env.initial_position], [initial_h], color="blue", label="Start")
        custom_handles.append(start_scatter)
        target_cross = ax.plot(self.env.target_position, target_h, marker='x', color='red', markersize=10, markeredgewidth=3, label='Target')[0]
        custom_handles.append(target_cross)

        # LQR linearization points if have
        for sim in simulators:

            if sim.controller.type == 'LQR' and not np.allclose(sim.controller.state_lin, sim.controller.target_state):
                lin_position = sim.controller.state_lin[0]
                lin_h = float(sim.env.h(lin_position).full().flatten()[0])
                lin_point = ax.plot(lin_position, lin_h, marker='v', color='orange', markersize=7, markeredgewidth=3, label='Linearization Point of LQR')[0]
                custom_handles.append(lin_point)

        # Car parameters
        car_length= 0.2
        wheel_radius = 0.035
        axle_offset = 1.5 * wheel_radius   # height from wheel center to car bottom
        car_shapes = {}
        car_bodies = {}
        wheel_lefts = {}
        wheel_rights = {}
        colors = [self.color] + self.color_list[:len(simulators)]

        # Self car
        car_shape_self, car_body_self, wheel_left_self, wheel_right_self = add_car(ax, car_length, wheel_radius, color=self.color, filled=False)

        # Simulators' cars
        for sim, color in zip(simulators, colors[1:]):
            if if_gray:
                car_shape, car_body, wheel_left, wheel_right = add_car(ax, car_length, wheel_radius, color='gray', filled=False)
            else:
                car_shape, car_body, wheel_left, wheel_right = add_car(ax, car_length, wheel_radius, color=self.color, filled=False)
            car_shapes[sim] = car_shape
            car_bodies[sim] = car_body
            wheel_lefts[sim] = wheel_left
            wheel_rights[sim] = wheel_right

        # Title using correct controller names
        controller_names = " vs. ".join(
            [self.simulator.controller_name] + [sim.controller_name for sim in simulators]
        )
        ax.set_title(controller_names)

        # Legends
        car_legend_handles = [
            mpatches.Patch(edgecolor=self.color, facecolor='none', linewidth=2, label=self.simulator.controller_name)
        ]
        for sim, color in zip(simulators, colors[1:]):
            if if_gray:
                car_legend_handles.append(mpatches.Patch(edgecolor='gray', facecolor='none', linewidth=2, label=sim.controller_name))
            else:
                car_legend_handles.append(mpatches.Patch(edgecolor=color, facecolor='none', linewidth=2, label=sim.controller_name))

        ax.legend(handles=custom_handles + car_legend_handles, loc='lower right')

        # Animation update function
        def update(frame):

            # Self
            current_position = self.position[frame]
            animate(self.env, current_position, car_length, wheel_radius, axle_offset, 
                    car_shape_self, car_body_self, wheel_left_self, wheel_right_self)

            # Simulators
            for sim in simulators:
                if sim is self:
                    continue
                frame_ref = int(frame*self.dt/sim.dt)  # Adjust frame index for each simulator
                current_position = sim.get_trajectories()[0][:, 0][frame_ref]
                animate(
                    sim.env,
                    current_position,
                    car_length, wheel_radius, axle_offset,
                    car_shapes[sim], 
                    car_bodies[sim],
                    wheel_lefts[sim],
                    wheel_rights[sim]
                )

        # Animate
        anim = FuncAnimation(fig, update, frames=len(self.t_eval),
                            interval=1000 / self.refresh_rate, repeat=False)
        
        # Save animation if required
        if if_save:
            anim.save("animation.mp4", writer="ffmpeg", fps=self.refresh_rate)
        
        plt.close(fig)
        
        return HTML(anim.to_jshtml())