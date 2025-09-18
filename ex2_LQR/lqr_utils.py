import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp
from scipy.optimize import minimize



from utils.env import Env, Dynamics
from utils.controller import BaseController, is_square, is_symmetric, is_positive_definite, is_positive_semi_definite, check_input_constraints








# Derived class for LQR Controller with finite horizon
class FiniteLQRController(BaseController):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Q_N: np.ndarray, 
            freq: float, 
            horizon: int,
            name: str = 'LQR_finite', 
            type: str = 'LQR_finite', 
            verbose: bool = False
        ) -> None:

        super().__init__(env, dynamics, freq, name, type, verbose)
        
        # Initialize as private property
        self._Q = None
        self._R = None
        self._Q_N = None

        self.N = horizon

        self.K_list = [None] * self.N  # LQR gain matrix
        
        # Call setter for the check and update the value of private property
        self.Q = Q
        self.R = R
        self.Q_N = Q_N

        self.A = None  # State transfer matrix
        self.B = None  # Input matrix

        self.x_eq = None  # Equilibrium state
        self.u_eq = None  # Equilibrium input

        self.state_lin = self.target_state
        
        self.setup()

    @property
    def Q(self) -> np.ndarray:
        return self._Q

    @Q.setter
    def Q(self, value: np.ndarray) -> None:

        is_square(value)
        is_symmetric(value)
        is_positive_semi_definite(value)

        if self.verbose:
            print("Check passed, Q is a symmetric, positive semi-definite matrix.")

        self._Q = value

    @property
    def Q_N(self) -> np.ndarray:
        return self._Q_N

    @Q_N.setter
    def Q_N(self, value: np.ndarray) -> None:

        is_square(value)
        is_symmetric(value)
        is_positive_semi_definite(value)

        if self.verbose:
            print("Check passed, Q_N is a symmetric, positive semi-definite matrix.")

        self._Q_N = value

    @property
    def R(self) -> np.ndarray:
        return self._R

    @R.setter
    def R(self, value: np.ndarray) -> None:

        is_square(value)
        is_symmetric(value)
        is_positive_definite(value)

        if self.verbose:
            print("Check passed, R is a symmetric, positive definite matrix.")
        
        self._R = value
    
    def setup(self) -> None:
        
        # Set up equilibrium state
        # Note that if target state is not on the slope, self.u_eq = 0 -> will not work for the nonlinear case
        self.x_eq = self.state_lin

        # Solve input at equilibrium
        self.u_eq = self.dynamics.get_equilibrium_input(self.x_eq)

        # Linearize dynamics at equilibrium
        self.A, self.B = self.dynamics.get_linearized_AB_discrete(
            current_state=self.x_eq, current_input=self.u_eq, dt=self.dt
        )

        # Initialize terminal cost
        P = self.Q_N.copy()

        # Solve Bellman Recursion from backwardsto compute gain matrix
        for k in reversed(range(self.N)):
            K = - np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
            self.K_list[k] = K
            P = self.Q + self.A.T @ P @ self.A + self.A.T @ P @ self.B @ K

        if self.verbose:
            print(f"LQR Gain Matrix K: {self.K}")

    @check_input_constraints
    def compute_action(
        self, 
        current_state: np.ndarray, 
        current_time: int
    ) -> np.ndarray:
        
        if any(k is None for k in self.K_list):
            raise ValueError("LQR gain matrix K is not computed. Call setup() first.")
        
        k = current_time  # assume current_time in [0, N-1]

        if k >= self.N:
            k = self.N - 1  # use terminal gain if past horizon

        # Compute state error
        det_x = current_state - self.target_state

        # Get the corresponding gain matrix for the current time step
        K_k = self.K_list[k]

        # Apply control law
        u = self.u_eq + K_k @ det_x

        return u


# Derived class for LQR Controller with infinite horizon
class LQRController(BaseController):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            freq: float, 
            name: str = 'LQR', 
            type: str = 'LQR', 
            verbose: bool = False
        ) -> None:

        super().__init__(env, dynamics, freq, name, type, verbose)
        
        # Initialize as private property
        self._Q = None
        self._R = None
        self._K = None  # LQR gain matrix
        
        # Call setter for the check and update the value of private property
        self.Q = Q
        self.R = R

        self.A = None  # State transfer matrix
        self.B = None  # Input matrix

        self.x_eq = None  # Equilibrium state
        self.u_eq = None  # Equilibrium input

        self.state_lin = self.target_state
        
        # need no external objects for setup, can directly call the setup function here
        if self.type in ['LQR', 'MPC']:
            self.setup()

    @property
    def Q(self) -> np.ndarray:
        return self._Q

    @Q.setter
    def Q(self, value: np.ndarray) -> None:

        is_square(value)
        is_symmetric(value)
        is_positive_semi_definite(value)

        if self.verbose:
            print("Check passed, Q is a symmetric, positive semi-definite matrix.")

        self._Q = value

    @property
    def R(self) -> np.ndarray:
        return self._R

    @R.setter
    def R(self, value: np.ndarray) -> None:

        is_square(value)
        is_symmetric(value)
        is_positive_definite(value)

        if self.verbose:
            print("Check passed, R is a symmetric, positive definite matrix.")
        
        self._R = value
    
    @property
    def K(self) -> np.ndarray:
        return self._K

    @K.setter
    def K(self, value: np.ndarray) -> None:

        eigvals = np.linalg.eigvals(self.A + self.B @ value)

        if np.any(np.abs(eigvals) > 1):
            raise ValueError("Warning: not all eigenvalue of A_cl inside unit circle, close-loop system is unstable!")
        
        elif self.verbose:
            print(f"Check passed, current gain K={value}, close-loop system is stable.")

        self._K = value
    
    def set_lin_point(
        self, 
        state_lin: np.ndarray
    ) -> None:
        
        """
        Set the linearization point and refresh the controller setup.
        """

        self.state_lin = state_lin
        
        # Refresh
        self.setup()

    def setup(self) -> None:
        
        # Set up equilibrium state
        # Note that if target state is not on the slope, self.u_eq = 0 -> will not work for the nonlinear case
        self.x_eq = self.state_lin

        # Solve input at equilibrium
        self.u_eq = self.dynamics.get_equilibrium_input(self.x_eq)

        # Linearize dynamics at equilibrium
        self.A, self.B = self.dynamics.get_linearized_AB_discrete(
            current_state=self.x_eq, current_input=self.u_eq, dt=self.dt
        )

        # Solve DARE to compute gain matrix
        P = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = - np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

        if self.verbose:
            print(f"LQR Gain Matrix K: {self.K}")

    @check_input_constraints
    def compute_action(
        self, 
        current_state: np.ndarray, 
        current_time: int
    ) -> np.ndarray:
        
        if self.K is None:
            raise ValueError("LQR gain matrix K is not computed. Call setup() first.")

        # Compute state error
        det_x = current_state - self.target_state

        # Apply control law
        u = self.u_eq + self.K @ det_x

        #print(f"self.u_eq: {self.u_eq}")
        #print(f"self.K: {self.K}")
        #print(f"det_x: {det_x}")
        #print(f"self.K @ det_x: {self.K @ det_x}")

        return u

