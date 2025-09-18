import numpy as np
import time
from abc import ABC, abstractmethod
from typing import List, Callable, Union, Optional, Tuple, Any
from functools import wraps

from utils.env import Env, Dynamics





'''------Decorators for Controllers------'''

def is_square(
        matrix: np.ndarray
    ) -> bool:

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Warning: must be a square matrix! ({matrix})")
    return True

def is_symmetric(
        matrix: np.ndarray
    ) -> bool:

    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"Warning: must be a symmetric matrix! ({matrix})")
    return True

def is_positive_semi_definite(
        matrix: np.ndarray
    ) -> bool:

    eigvals = np.linalg.eigvals(matrix)
    if not np.all(np.greater_equal(eigvals, 0)):
        raise ValueError(f"Warning: must be a positive semi-definite matrix! ({matrix})")
    return True

def is_positive_definite(
        matrix: np.ndarray
    ) -> bool:

    eigvals = np.linalg.eigvals(matrix)
    if not np.all(np.greater(eigvals, 0)):
        raise ValueError(f"Warning: must be a positive definite matrix! ({matrix})")
    return True

def check_input_constraints(
        compute_action, 
        mode='clipping'
    ) -> Callable:

    """
    Decorator, for checking whether input is compatible with given constraints
    input constraint get from self.env
    """

    @wraps(compute_action)
    def wrapper(self, current_state, current_time):

        # Get upper and lower bound of input from env
        input_lbs = self.env.input_lbs
        input_ubs = self.env.input_ubs

        # Call original method 'compute_action'
        result = compute_action(self, current_state, current_time)
        u = result if not isinstance(result, tuple) else result[0] # incase mpc (contains predicted trajs)

        # Check whether input is compatible with given constraints
        if input_lbs is not None:

            if not np.all(input_lbs <= u):
                if mode == 'raise_error':
                    raise ValueError(f"Warning: raw control input u={u} is beyond the lower limit {input_lbs}!")
                elif mode == 'clipping':
                    print(f"Warning: raw control input u={u} is beyond the lower limit {input_lbs}! Clip to lower limit u={input_lbs}.")
                    u = np.array([input_lbs])

        if input_ubs is not None:
            if not np.all(u <= input_ubs):
                if mode == 'raise_error':
                    raise ValueError(f"Warning: control input u={u} is beyond the upper limit {input_ubs}!")
                elif mode == 'clipping':
                    print(f"Warning: raw control input u={u} is beyond the upper limit {input_ubs}! Clip to upper limit u={input_ubs}.")
                    u = np.array([input_ubs])
        
        if isinstance(result, tuple):
            return (u,) + result[1:]
        else:
            return u
    
    return wrapper

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__qualname__} took {end - start:.6f} seconds")
        return result
    return wrapper






'''------Base Controller Class------'''

class BaseController(ABC):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            freq: float, 
            name: str, 
            type: str,
            verbose: bool = False
        ) -> None:

        self.name = name
        self.type = type

        self.env = env
        self.dynamics = dynamics

        self.freq = freq
        self.dt = 1 / self.freq

        self.verbose = verbose

        self.dim_states = self.dynamics.dim_states
        self.dim_inputs = self.dynamics.dim_inputs

        self.init_state = self.env.init_state
        self.target_state = self.env.target_state

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize necessary matrices or parameters for the controller.
        """
        pass

    @abstractmethod
    def compute_action(self, current_state: np.ndarray, current_time) -> np.ndarray:
        """
        Compute control action based on the current state and time.
        """
        pass



