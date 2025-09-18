import numpy as np
import casadi as ca
import time
import matplotlib.pyplot as plt
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

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


from utils.env import Env, Dynamics
from utils.controller import check_input_constraints
from ex2_LQR.lqr_utils import LQRController





'''------Open-Loop Controllers------'''

# Class for linear OCP Controller with 2-norm cost (open-loop)
#   Note: 1) assumes linearized dynamics, 2-norm cost, open-loop control, stabilization task
#         2) implementation based on cvxopt QP solver, no symbolic representation
class LinearOCPController:
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Qf: np.ndarray, 
            freq: float, 
            N: int, 
            name='OCP-2-norm', 
            type='OCP', 
            verbose=False
        ) -> None:

        self.env = env
        self.dynamics = dynamics
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.freq = freq
        self.dt = 1.0 / freq
        self.N = N
        self.name = name
        self.type = type
        self.verbose = verbose

        self.dim_states = dynamics.dim_states
        self.dim_inputs = dynamics.dim_inputs

        self.u_seq = None

    def compute_action(
        self, 
        current_state: np.ndarray, 
        current_time: int
    ) -> np.ndarray:
    
        if self.u_seq is None:

            start_time = time.time()

            self._solve_ocp(current_state)

            print(f"Computation time: {time.time()-start_time}")

        index = current_time
        if index < self.u_seq.shape[0]:
            return self.u_seq[index]
        else:
            return self.u_seq[-1]

    def _solve_ocp(
        self, 
        current_state: np.ndarray
    ) -> None:
        
        x0 = current_state.reshape(-1, 1)
        x_ref = self.env.target_state.reshape(-1, 1)
        u_ref = np.array(self.dynamics.get_equilibrium_input(x_ref)).reshape(-1, 1)

        A_d, B_d = self.dynamics.get_linearized_AB_discrete(x0, u_ref, self.dt)

        nx = self.dim_states
        nu = self.dim_inputs
        N = self.N
        n_vars = (N + 1) * nx + N * nu

        # Cost
        H = np.zeros((n_vars, n_vars))
        f = np.zeros((n_vars, 1))

        for k in range(N):
            idx_x = slice(k * nx, (k + 1) * nx)
            idx_u = slice((N + 1) * nx + k * nu, (N + 1) * nx + (k + 1) * nu)
            H[idx_x, idx_x] = self.Q
            f[idx_x] = -self.Q @ x_ref
            H[idx_u, idx_u] = self.R
            f[idx_u] = -self.R @ u_ref

        idx_terminal = slice(N * nx, (N + 1) * nx)
        H[idx_terminal, idx_terminal] = self.Qf
        f[idx_terminal] = -self.Qf @ x_ref

        # Dynamics constraints
        Aeq = []
        beq = []

        for k in range(N):
            row = np.zeros((nx, n_vars))
            idx_xk = slice(k * nx, (k + 1) * nx)
            idx_xkp1 = slice((k + 1) * nx, (k + 2) * nx)
            idx_uk = slice((N + 1) * nx + k * nu, (N + 1) * nx + (k + 1) * nu)
            row[:, idx_xk] = A_d
            row[:, idx_uk] = B_d
            row[:, idx_xkp1] = -np.eye(nx)
            Aeq.append(row)
            beq.append(np.zeros((nx, 1)))

        row0 = np.zeros((nx, n_vars))
        row0[:, 0:nx] = np.eye(nx)
        Aeq.insert(0, row0)
        beq.insert(0, x0)

        Aeq = np.vstack(Aeq)
        beq = np.vstack(beq)

        # Inequality constraints
        G = []
        h = []

        for k in range(N + 1):
            idx_x = slice(k * nx, (k + 1) * nx)
            if self.env.state_lbs is not None:
                Gx_l = np.zeros((nx, n_vars))
                Gx_l[:, idx_x] = -np.eye(nx)
                G.append(Gx_l)
                h.append(-np.array(self.env.state_lbs).reshape(-1, 1))
            if self.env.state_ubs is not None:
                Gx_u = np.zeros((nx, n_vars))
                Gx_u[:, idx_x] = np.eye(nx)
                G.append(Gx_u)
                h.append(np.array(self.env.state_ubs).reshape(-1, 1))

        for k in range(N):
            idx_u = slice((N + 1) * nx + k * nu, (N + 1) * nx + (k + 1) * nu)
            if self.env.input_lbs is not None:
                Gu_l = np.zeros((nu, n_vars))
                Gu_l[:, idx_u] = -np.eye(nu)
                G.append(Gu_l)
                h.append(-np.array(self.env.input_lbs).reshape(-1, 1))
            if self.env.input_ubs is not None:
                Gu_u = np.zeros((nu, n_vars))
                Gu_u[:, idx_u] = np.eye(nu)
                G.append(Gu_u)
                h.append(np.array(self.env.input_ubs).reshape(-1, 1))

        G = np.vstack(G) if G else np.zeros((1, n_vars))
        h = np.vstack(h) if h else np.array([[1e10]])

        solvers.options['show_progress'] = self.verbose
        sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h), matrix(Aeq), matrix(beq))
        z_opt = np.array(sol['x']).flatten()

        u_start = (N + 1) * nx
        self.u_seq = z_opt[u_start:].reshape(N, nu) + u_ref


# Class for linear OCP Controller with 1-norm cost (open-loop)
#   Note: 1) assumes linearized dynamics, 1-norm cost, open-loop control, stabilization task
#         2) implementation based on cvxopt QP solver, no symbolic representation
class LinearOCP1NormController:
    def __init__(
        self, 
        env: Env, 
        dynamics: Dynamics, 
        freq: float, 
        N: int, 
        name='OCP-1-norm', 
        type='OCP', 
        verbose=False
    ) -> None:
        
        self.env = env
        self.dynamics = dynamics
        self.freq = freq
        self.dt = 1.0 / freq
        self.N = N
        self.name = name
        self.type = type
        self.verbose = verbose

        self.dim_states = dynamics.dim_states
        self.dim_inputs = dynamics.dim_inputs

        self.u_seq = None  # open-loop control sequence

    def compute_action(
        self, 
        current_state: np.ndarray, 
        current_time: int
    ) -> np.ndarray:
        
        if self.u_seq is None:
            self._solve_ocp(current_state)

        index = current_time
        if index < self.u_seq.shape[0]:
            return self.u_seq[index]
        else:
            return self.u_seq[-1]

    def _solve_ocp(
        self, 
        current_state: np.ndarray
    ) -> None:
        
        x0 = current_state.reshape(-1, 1)
        u0 = np.zeros((self.dim_inputs, 1))
        A_d, B_d = self.dynamics.get_linearized_AB_discrete(x0, u0, self.dt)

        nx = self.dim_states
        nu = self.dim_inputs
        N = self.N
        n_vars = nx * (N + 1) + nu * N + nu * N  # x, u, t

        def idx_x(k): return slice(k * nx, (k + 1) * nx)
        def idx_u(k): return slice((N + 1) * nx + k * nu, (N + 1) * nx + (k + 1) * nu)
        def idx_t(k): return slice((N + 1) * nx + N * nu + k * nu, (N + 1) * nx + N * nu + (k + 1) * nu)

        # Cost: sum of slack variables t_k
        H = np.zeros((n_vars, n_vars))
        f = np.zeros((n_vars, 1))
        for k in range(N):
            f[idx_t(k)] = 1.0

        # Equality constraints
        Aeq = []
        beq = []

        for k in range(N):
            row = np.zeros((nx, n_vars))
            row[:, idx_x(k)] = A_d
            row[:, idx_u(k)] = B_d
            row[:, idx_x(k+1)] = -np.eye(nx)
            Aeq.append(row)
            beq.append(np.zeros((nx, 1)))

        row0 = np.zeros((nx, n_vars))
        row0[:, idx_x(0)] = np.eye(nx)
        Aeq.insert(0, row0)
        beq.insert(0, x0)

        xg = self.env.target_state.reshape(-1, 1)
        rowT = np.zeros((nx, n_vars))
        rowT[:, idx_x(N)] = np.eye(nx)
        Aeq.append(rowT)
        beq.append(xg)

        Aeq = np.vstack(Aeq)
        beq = np.vstack(beq)

        # Inequality constraints
        G = []
        h = []

        for k in range(N):
            G1 = np.zeros((nu, n_vars))
            G1[:, idx_u(k)] = np.eye(nu)
            G1[:, idx_t(k)] = -np.eye(nu)
            G.append(G1)
            h.append(np.zeros((nu, 1)))

            G2 = np.zeros((nu, n_vars))
            G2[:, idx_u(k)] = -np.eye(nu)
            G2[:, idx_t(k)] = -np.eye(nu)
            G.append(G2)
            h.append(np.zeros((nu, 1)))

            if self.env.input_ubs is not None:
                G3 = np.zeros((nu, n_vars))
                G3[:, idx_u(k)] = np.eye(nu)
                G.append(G3)
                h.append(np.array(self.env.input_ubs).reshape(-1, 1))

            if self.env.input_lbs is not None:
                G4 = np.zeros((nu, n_vars))
                G4[:, idx_u(k)] = -np.eye(nu)
                G.append(G4)
                h.append(-np.array(self.env.input_lbs).reshape(-1, 1))

        G = np.vstack(G)
        h = np.vstack(h)

        # Solve QP
        solvers.options['show_progress'] = self.verbose
        sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h), matrix(Aeq), matrix(beq))
        z_opt = np.array(sol['x']).flatten()

        u_start = (N + 1) * nx
        self.u_seq = z_opt[u_start:u_start + N * nu].reshape(N, nu)

    
# Class for linear OCP Controller with inf-norm cost (open-loop)
#   Note: 1) assumes linearized dynamics, inf-norm cost, open-loop control, stabilization task
#         2) implementation based on cvxopt QP solver, no symbolic representation
class LinearOCPInfNormController:
    def __init__(
        self, 
        env: Env, 
        dynamics: Dynamics, 
        freq: float, 
        N: int,
        name: str = 'OCP-inf-norm', 
        type: str = 'OCP', 
        verbose: bool = False
    ) -> None:

        self.env = env
        self.dynamics = dynamics
        self.N = N
        self.name = name
        self.type = type
        self.freq = freq
        self.dt = 1.0 / freq
        self.verbose = verbose

        self.dim_states = dynamics.dim_states
        self.dim_inputs = dynamics.dim_inputs

        self.u_seq = None  # store open-loop sequence

    @check_input_constraints
    def compute_action(
        self, 
        current_state: np.ndarray, 
        current_time: int
    ) -> np.ndarray:
        
        if self.u_seq is None:
            self._solve_ocp(current_state)

        if current_time < len(self.u_seq):
            return self.u_seq[current_time]
        else:
            return self.u_seq[-1]  # repeat last action if time exceeds

    def _solve_ocp(
        self, 
        current_state: np.ndarray
    ) -> None:
        
        x0 = current_state.reshape(-1, 1)
        u0 = np.zeros((self.dim_inputs, 1))

        A_d, B_d = self.dynamics.get_linearized_AB_discrete(x0, u0, self.dt)

        n_vars = self.dim_states * (self.N + 1) + self.dim_inputs * self.N + 1
        def idx_x(k): return slice(k * self.dim_states, (k + 1) * self.dim_states)
        def idx_u(k): return slice((self.N + 1) * self.dim_states + k * self.dim_inputs,
                                   (self.N + 1) * self.dim_states + (k + 1) * self.dim_inputs)
        idx_t = n_vars - 1

        H = np.zeros((n_vars, n_vars))
        f = np.zeros((n_vars, 1))
        f[idx_t] = 1.0

        Aeq, beq = [], []

        for k in range(self.N):
            row = np.zeros((self.dim_states, n_vars))
            row[:, idx_x(k)] = A_d
            row[:, idx_u(k)] = B_d
            row[:, idx_x(k + 1)] = -np.eye(self.dim_states)
            Aeq.append(row)
            beq.append(np.zeros((self.dim_states, 1)))

        row0 = np.zeros((self.dim_states, n_vars))
        row0[:, idx_x(0)] = np.eye(self.dim_states)
        Aeq.insert(0, row0)
        beq.insert(0, x0)

        rowT = np.zeros((self.dim_states, n_vars))
        rowT[:, idx_x(self.N)] = np.eye(self.dim_states)
        Aeq.append(rowT)
        xg = self.env.target_state.reshape(-1, 1)
        beq.append(xg)

        Aeq = np.vstack(Aeq)
        beq = np.vstack(beq)

        G, h = [], []

        for k in range(self.N):
            for i in range(self.dim_inputs):
                row1 = np.zeros((1, n_vars))
                row1[0, idx_u(k).start + i] = 1.0
                row1[0, idx_t] = -1.0
                G.append(row1)
                h.append([0.0])

                row2 = np.zeros((1, n_vars))
                row2[0, idx_u(k).start + i] = -1.0
                row2[0, idx_t] = -1.0
                G.append(row2)
                h.append([0.0])

            if self.env.input_ubs is not None:
                row = np.zeros((self.dim_inputs, n_vars))
                row[:, idx_u(k)] = np.eye(self.dim_inputs)
                G.append(row)
                h.append(np.array(self.env.input_ubs).reshape(-1, 1))

            if self.env.input_lbs is not None:
                row = np.zeros((self.dim_inputs, n_vars))
                row[:, idx_u(k)] = -np.eye(self.dim_inputs)
                G.append(row)
                h.append(-np.array(self.env.input_lbs).reshape(-1, 1))

        G = np.vstack(G)
        h = np.vstack(h)

        solvers.options['show_progress'] = self.verbose
        sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h), matrix(Aeq), matrix(beq))
        z_opt = np.array(sol['x']).flatten()

        u_start = (self.N + 1) * self.dim_states
        self.u_seq = z_opt[u_start:u_start + self.N * self.dim_inputs].reshape(self.N, self.dim_inputs)

        if self.verbose:
            print(f"[open-loop inf-norm OCP] Optimal u_seq:\n{self.u_seq}")
    





'''------Model Predictive Control (MPC)------'''


# Class for linear MPC Controller with 2-norm cost (closed-loop)
#   Note: 1) assumes linearized dynamics, 2-norm cost, closed-loop control, stabilization task
#         2) implementation based on cvxopt QP solver, no symbolic representation
class LinearMPCController:
    def __init__(self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Qf: np.ndarray, 
            freq: float, 
            N: int, 
            name: str = 'LMPC', 
            type: str = 'MPC', 
            verbose: bool = False
        ) -> None:

        self.env = env
        self.dynamics = dynamics

        self.Q = Q
        self.R = R
        self.Qf = Qf

        self.N = N

        self.name = name
        self.type = type

        self.freq = freq
        self.dt = 1.0 / freq
        
        self.verbose = verbose

        self.dim_states = dynamics.dim_states
        self.dim_inputs = dynamics.dim_inputs

        self.x_pred = None
        self.u_pred = None
    
    @check_input_constraints
    def compute_action(
        self, 
        current_state: np.ndarray, 
        current_time: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        x0 = current_state.reshape(-1, 1)
        u0 = np.zeros((self.dim_inputs, 1))

        A_d, B_d = self.dynamics.get_linearized_AB_discrete(x0, u0, self.dt)

        n_vars = self.dim_states * (self.N + 1) + self.dim_inputs * self.N

        # Cost
        H = np.zeros((n_vars, n_vars))
        f = np.zeros((n_vars, 1))

        x_ref = self.env.target_state.reshape(-1, 1)  # assumed constant
        u_ref = np.array(self.dynamics.get_equilibrium_input(x_ref)).reshape(-1, 1)        # assumed constant

        for k in range(self.N):
            idx_x = slice(k * self.dim_states, (k + 1) * self.dim_states)
            idx_u = slice((self.N + 1) * self.dim_states + k * self.dim_inputs,
                          (self.N + 1) * self.dim_states + (k + 1) * self.dim_inputs)

            Qk = self.Q
            H[idx_x, idx_x] = Qk
            H[idx_u, idx_u] = self.R

            f[idx_x] = -Qk @ x_ref
            f[idx_u] = -self.R @ u_ref

        # Terminal cost
        idx_x_terminal = slice(self.N * self.dim_states, (self.N + 1) * self.dim_states)
        H[idx_x_terminal, idx_x_terminal] = self.Qf
        f[idx_x_terminal] = -self.Qf @ x_ref

        # Dynamics constraints
        Aeq = []
        beq = []

        for k in range(self.N):
            row = np.zeros((self.dim_states, n_vars))
            idx_xk = slice(k * self.dim_states, (k + 1) * self.dim_states)
            idx_xk_next = slice((k + 1) * self.dim_states, (k + 2) * self.dim_states)
            idx_uk = slice((self.N + 1) * self.dim_states + k * self.dim_inputs,
                           (self.N + 1) * self.dim_states + (k + 1) * self.dim_inputs)
            row[:, idx_xk_next] = -np.eye(self.dim_states)
            row[:, idx_xk] = A_d
            row[:, idx_uk] = B_d
            Aeq.append(row)
            beq.append(np.zeros((self.dim_states, 1)))

        # Add x0 = current_state as hard equality constraint
        row0 = np.zeros((self.dim_states, n_vars))
        row0[:, :self.dim_states] = np.eye(self.dim_states)
        Aeq.insert(0, row0)
        beq.insert(0, x0)

        Aeq = np.vstack(Aeq)
        beq = np.vstack(beq)

        # Inequality constraints
        G_list = []
        h_list = []

        for k in range(self.N + 1):
            # State constraints
            if self.env.state_lbs is not None:
                Gx_l = np.zeros((self.dim_states, n_vars))
                Gx_l[:, k * self.dim_states:(k + 1) * self.dim_states] = -np.eye(self.dim_states)
                G_list.append(Gx_l)
                h_list.append(-np.array(self.env.state_lbs).reshape(-1, 1))
            if self.env.state_ubs is not None:
                Gx_u = np.zeros((self.dim_states, n_vars))
                Gx_u[:, k * self.dim_states:(k + 1) * self.dim_states] = np.eye(self.dim_states)
                G_list.append(Gx_u)
                h_list.append(np.array(self.env.state_ubs).reshape(-1, 1))

        for k in range(self.N):
            # Input constraints
            if self.env.input_lbs is not None:
                Gu_l = np.zeros((self.dim_inputs, n_vars))
                Gu_l[:, (self.N + 1) * self.dim_states + k * self.dim_inputs:
                          (self.N + 1) * self.dim_states + (k + 1) * self.dim_inputs] = -np.eye(self.dim_inputs)
                G_list.append(Gu_l)
                h_list.append(-np.array(self.env.input_lbs).reshape(-1, 1))
            if self.env.input_ubs is not None:
                Gu_u = np.zeros((self.dim_inputs, n_vars))
                Gu_u[:, (self.N + 1) * self.dim_states + k * self.dim_inputs:
                          (self.N + 1) * self.dim_states + (k + 1) * self.dim_inputs] = np.eye(self.dim_inputs)
                G_list.append(Gu_u)
                h_list.append(np.array(self.env.input_ubs).reshape(-1, 1))

        # Check if there are any constraints
        if len(G_list) > 0:
            G = np.vstack(G_list)
            h = np.vstack(h_list)
        else:
            G = np.zeros((1, n_vars))
            h = np.array([[1e10]])

        # Solve the QP
        solvers.options['show_progress'] = False
        sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h), matrix(Aeq), matrix(beq))
        z_opt = np.array(sol['x']).flatten()

        x_opt = z_opt[: (self.N + 1) * self.dim_states].reshape(self.N + 1, self.dim_states).T
        u_opt = z_opt[(self.N + 1) * self.dim_states:].reshape(self.N, self.dim_inputs).T

        if self.verbose:
            print(f"Optimal control action: {u_opt[:, 0]}")
            print(f"Predicted x: {x_opt}")
            print(f"Predicted u: {u_opt}")

        self.x_pred = x_opt
        self.u_pred = u_opt

        return u_opt[:, 0].flatten()+u_ref, x_opt.T, u_opt


# Derived class for MPC Controller
#   Note: 1) any dynamics (linear/nonlinear), 2-norm cost, closed-loop control, stabilization task
#         2) implementation based on CasADi symbolic system and Acados
class MPCController(LQRController):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Qf: np.ndarray, 
            freq: float, 
            N: int, 
            name: str = 'MPC', 
            type: str = 'MPC', 
            verbose: bool = True
        ) -> None:

        """
        Initialize the MPC Controller with Acados.

        Args:
        - env: The environment providing initial and target states.
        - dynamics: The system dynamics.
        - Q: State cost matrix.
        - R: Control cost matrix.
        - Qf: Terminal state cost matrix.
        - freq: Control frequency.
        - N: Prediction horizon.
        - verbose: Print debug information if True.
        """

        self.Qf = Qf

        self.N = N  # Prediction horizon

        self.ocp = None
        self.solver = None

        super().__init__(env, dynamics, Q, R, freq, name, type, verbose)

    def setup(self) -> None:
        """
        Define the MPC optimization problem using Acados.
        """
        
        ## Model
        # Set up Acados model
        model = AcadosModel()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model.name = f"{self.name}_{timestamp}"


        # Define model: x_dot = f(x, u)
        model.x = self.dynamics.states
        model.u = self.dynamics.inputs
        model.f_expl_expr = ca.vertcat(self.dynamics.dynamics_function(self.dynamics.states, self.dynamics.inputs))
        model.f_impl_expr = None # no needed, we already have the explicit model


        ## Optimal control problem
        # Set up Acados OCP
        ocp = AcadosOcp()
        ocp.model = model # link to the model (class: AcadosModel)
        ocp.dims.N = self.N  # prediction horizon
        ocp.solver_options.tf = self.N * self.dt  # total prediction time
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" # Partially condensing interior-point method
        ocp.solver_options.integrator_type = "ERK" # explicit Runge-Kutta
        ocp.solver_options.nlp_solver_type = "SQP" # sequential quadratic programming

        # Set up other hyperparameters in SQP solving
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.nlp_solver_tol_stat = 1E-6
        ocp.solver_options.nlp_solver_tol_eq = 1E-6
        ocp.solver_options.nlp_solver_tol_ineq = 1E-6
        ocp.solver_options.nlp_solver_tol_comp = 1E-6
        
        # For debugging
        #ocp.solver_options.print_level = 2

        # Set up cost function
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = np.block([
            [self.Q, np.zeros((self.dim_states, self.dim_inputs))],
            [np.zeros((self.dim_inputs, self.dim_states)), self.R],
        ])
        ocp.cost.W_e = self.Qf

        # Set up mapping from QP to OCP
        # Define output matrix for non-terminal state
        ocp.cost.Vx = np.block([
            [np.eye(self.dim_states)],
            [np.zeros((self.dim_inputs, self.dim_states))]
        ])
        # Define breakthrough matrix for non-terminal state
        ocp.cost.Vu = np.block([
            [np.zeros((self.dim_states, self.dim_inputs))],
            [np.eye(self.dim_inputs)]
        ])
        # Define output matrix for terminal state
        ocp.cost.Vx_e = np.eye(self.dim_states)

        # Initialize reference of task (stabilization)
        ocp.cost.yref = np.zeros(self.dim_states + self.dim_inputs) 
        ocp.cost.yref_e = np.zeros(self.dim_states) 

        # Define constraints
        ocp.constraints.x0 = self.init_state  # Initial state

        # State constraints
        ocp.constraints.idxbx = np.arange(self.dim_states)
        ocp.constraints.idxbx_e = np.arange(self.dim_states)
        if self.env.state_lbs is None and self.env.state_ubs is None:
            ocp.constraints.lbx_0 = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_0 = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx_e = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_e = np.full(self.dim_states, 1e6)
        elif self.env.state_lbs is not None and self.env.state_ubs is None:
            ocp.constraints.lbx_0 = np.array(self.env.state_lbs)
            ocp.constraints.ubx_0 = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx = np.array(self.env.state_lbs)
            ocp.constraints.ubx = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx_e = np.array(self.env.state_lbs)
            ocp.constraints.ubx_e = np.full(self.dim_states, 1e6)
        elif self.env.state_lbs is None and self.env.state_ubs is not None:
            ocp.constraints.lbx_0 = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_0 = np.array(self.env.state_ubs)
            ocp.constraints.lbx = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx = np.array(self.env.state_ubs)
            ocp.constraints.lbx_e = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_e = np.array(self.env.state_ubs)
        else:
            ocp.constraints.lbx_0 = np.array(self.env.state_lbs)
            ocp.constraints.ubx_0 = np.array(self.env.state_ubs)
            ocp.constraints.lbx = np.array(self.env.state_lbs)
            ocp.constraints.ubx = np.array(self.env.state_ubs)
            ocp.constraints.lbx_e = np.array(self.env.state_lbs)
            ocp.constraints.ubx_e = np.array(self.env.state_ubs)
        
        # Input constraints
        ocp.constraints.idxbu = np.arange(self.dim_inputs)
        if self.env.input_lbs is None and self.env.input_ubs is None:
            ocp.constraints.lbu = np.full(self.dim_inputs, -1e6)
            ocp.constraints.ubu = np.full(self.dim_inputs, 1e6)
        elif self.env.input_lbs is not None and self.env.input_ubs is None:
            ocp.constraints.lbu = np.array(self.env.input_lbs)
            ocp.constraints.ubu = np.full(self.dim_inputs, 1e6)
        elif self.env.input_lbs is None and self.env.input_ubs is not None:
            ocp.constraints.lbu = np.full(self.dim_inputs, -1e6)
            ocp.constraints.ubu = np.array(self.env.input_ubs)
        else:
            ocp.constraints.lbu = np.array(self.env.input_lbs)
            ocp.constraints.ubu = np.array(self.env.input_ubs)


        ## Ocp Solver
        # Set up Acados solver
        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file=f"{model.name}.json")

        if self.verbose:
            print("MPC setup with Acados completed.")

    @check_input_constraints
    def compute_action(
        self, 
        current_state: np.ndarray, 
        current_time: float = None
    ) -> np.ndarray:
        
        """
        Solve the MPC problem and compute the optimal control action.

        Args:
        - current_state: The current state of the system.
        - current_time: The current time (not used in this time-invariant case).

        Returns:
        - Optimal control action.
        """

        # Update initial state in the solver
        self.solver.set(0, "lbx", current_state)
        self.solver.set(0, "ubx", current_state)

        # Update reference trajectory for all prediction steps
        state_ref = self.target_state
        input_ref = np.zeros(self.dim_inputs)
        for i in range(self.N):
            self.solver.set(i, "yref", np.concatenate((state_ref, input_ref)))
        self.solver.set(self.N, "yref", state_ref) # set reference valur for y_N seperately (different shape)

        # Solve the MPC problem
        status = self.solver.solve()
        #if status != 0:
        #    raise ValueError(f"Acados solver failed with status {status}")

        # Extract the first control action
        u_optimal = self.solver.get(0, "u")

        # Extract the predictions
        x_pred = np.zeros((self.N + 1, self.dim_states))
        u_pred = np.zeros((self.N, self.dim_inputs))
        for i in range(self.N + 1):
            x_pred[i, :] = self.solver.get(i, "x")
            if i < self.N:
                u_pred[i, :] = self.solver.get(i, "u")

        if self.verbose:
            print(f"Optimal control action: {u_optimal}")
            print(f"x_pred: {x_pred}")
            print(f"u_pred: {u_pred}")

        return u_optimal, x_pred, u_pred


# Derived class for MPC Controller
#   Note: 1) any dynamics (linear/nonlinear), 2-norm cost, closed-loop control, tracking task
#         2) implementation based on CasADi symbolic system and Acados
class TrackingMPCController(LQRController):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Qf: np.ndarray, 
            freq: float, 
            N: int, 
            traj_ref: np.ndarray = None,
            name: str = 'MPC', 
            type: str = 'MPC', 
            verbose: bool = True
        ) -> None:

        """
        Initialize the MPC Controller with Acados.

        Args:
        - env: The environment providing initial and target states.
        - dynamics: The system dynamics.
        - Q: State cost matrix.
        - R: Control cost matrix.
        - Qf: Terminal state cost matrix.
        - freq: Control frequency.
        - N: Prediction horizon.
        - verbose: Print debug information if True.
        """

        self.Qf = Qf

        self.N = N  # Prediction horizon

        self.ocp = None
        self.solver = None
        self.traj_ref = traj_ref

        super().__init__(env, dynamics, Q, R, freq, name, type, verbose)

    def setup(self) -> None:
        """
        Define the MPC optimization problem using Acados.
        """
        
        ## Model
        # Set up Acados model
        model = AcadosModel()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model.name = f"{self.name}_{timestamp}"

        # Define model: x_dot = f(x, u)
        model.x = self.dynamics.states
        model.u = self.dynamics.inputs
        model.f_expl_expr = ca.vertcat(self.dynamics.dynamics_function(self.dynamics.states, self.dynamics.inputs))
        model.f_impl_expr = None # no needed, we already have the explicit model


        ## Optimal control problem
        # Set up Acados OCP
        ocp = AcadosOcp()
        ocp.model = model # link to the model (class: AcadosModel)
        ocp.dims.N = self.N  # prediction horizon
        ocp.solver_options.tf = self.N * self.dt  # total prediction time
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" # Partially condensing interior-point method
        ocp.solver_options.integrator_type = "ERK" # explicit Runge-Kutta
        ocp.solver_options.nlp_solver_type = "SQP" # sequential quadratic programming

        # Set up other hyperparameters in SQP solving
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.nlp_solver_tol_stat = 1E-6
        ocp.solver_options.nlp_solver_tol_eq = 1E-6
        ocp.solver_options.nlp_solver_tol_ineq = 1E-6
        ocp.solver_options.nlp_solver_tol_comp = 1E-6
        
        # For debugging
        #ocp.solver_options.print_level = 2

        # Set up cost function
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = np.block([
            [self.Q, np.zeros((self.dim_states, self.dim_inputs))],
            [np.zeros((self.dim_inputs, self.dim_states)), self.R],
        ])
        ocp.cost.W_e = self.Qf

        # Set up mapping from QP to OCP
        # Define output matrix for non-terminal state
        ocp.cost.Vx = np.block([
            [np.eye(self.dim_states)],
            [np.zeros((self.dim_inputs, self.dim_states))]
        ])
        # Define breakthrough matrix for non-terminal state
        ocp.cost.Vu = np.block([
            [np.zeros((self.dim_states, self.dim_inputs))],
            [np.eye(self.dim_inputs)]
        ])
        # Define output matrix for terminal state
        ocp.cost.Vx_e = np.eye(self.dim_states)

        # Initialize reference of task (stabilization)
        ocp.cost.yref = np.zeros(self.dim_states + self.dim_inputs) 
        ocp.cost.yref_e = np.zeros(self.dim_states) 

        # Define constraints
        ocp.constraints.x0 = self.init_state  # Initial state

        # State constraints
        ocp.constraints.idxbx = np.arange(self.dim_states)
        ocp.constraints.idxbx_e = np.arange(self.dim_states)
        if self.env.state_lbs is None and self.env.state_ubs is None:
            ocp.constraints.lbx_0 = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_0 = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx_e = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_e = np.full(self.dim_states, 1e6)
        elif self.env.state_lbs is not None and self.env.state_ubs is None:
            ocp.constraints.lbx_0 = np.array(self.env.state_lbs)
            ocp.constraints.ubx_0 = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx = np.array(self.env.state_lbs)
            ocp.constraints.ubx = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx_e = np.array(self.env.state_lbs)
            ocp.constraints.ubx_e = np.full(self.dim_states, 1e6)
        elif self.env.state_lbs is None and self.env.state_ubs is not None:
            ocp.constraints.lbx_0 = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_0 = np.array(self.env.state_ubs)
            ocp.constraints.lbx = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx = np.array(self.env.state_ubs)
            ocp.constraints.lbx_e = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_e = np.array(self.env.state_ubs)
        else:
            ocp.constraints.lbx_0 = np.array(self.env.state_lbs)
            ocp.constraints.ubx_0 = np.array(self.env.state_ubs)
            ocp.constraints.lbx = np.array(self.env.state_lbs)
            ocp.constraints.ubx = np.array(self.env.state_ubs)
            ocp.constraints.lbx_e = np.array(self.env.state_lbs)
            ocp.constraints.ubx_e = np.array(self.env.state_ubs)
        
        # Input constraints
        ocp.constraints.idxbu = np.arange(self.dim_inputs)
        if self.env.input_lbs is None and self.env.input_ubs is None:
            ocp.constraints.lbu = np.full(self.dim_inputs, -1e6)
            ocp.constraints.ubu = np.full(self.dim_inputs, 1e6)
        elif self.env.input_lbs is not None and self.env.input_ubs is None:
            ocp.constraints.lbu = np.array(self.env.input_lbs)
            ocp.constraints.ubu = np.full(self.dim_inputs, 1e6)
        elif self.env.input_lbs is None and self.env.input_ubs is not None:
            ocp.constraints.lbu = np.full(self.dim_inputs, -1e6)
            ocp.constraints.ubu = np.array(self.env.input_ubs)
        else:
            ocp.constraints.lbu = np.array(self.env.input_lbs)
            ocp.constraints.ubu = np.array(self.env.input_ubs)


        ## Ocp Solver
        # Set up Acados solver
        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file=f"{model.name}.json")

        if self.verbose:
            print("MPC setup with Acados completed.")

    @check_input_constraints
    def compute_action(
        self, 
        current_state: np.ndarray, 
        current_time: float = None
    ) -> np.ndarray:
        
        """
        Solve the MPC problem and compute the optimal control action.

        Args:
        - current_state: The current state of the system.
        - current_time: The current time (not used in this time-invariant case).

        Returns:
        - Optimal control action.
        """

        # Update initial state in the solver
        self.solver.set(0, "lbx", current_state)
        self.solver.set(0, "ubx", current_state)

        # Update reference trajectory for all prediction steps
        input_ref = np.zeros(self.dim_inputs)
        ref_length = self.traj_ref.shape[0]

        for i in range(self.N):
            index = min(current_time + i, ref_length - 1)
            state_ref = self.traj_ref[index, :self.dim_states]
            self.solver.set(i, "yref", np.concatenate((state_ref, input_ref)))
        index = min(current_time + self.N, ref_length - 1)
        terminal_state_ref = self.traj_ref[index, :self.dim_states]
        self.solver.set(self.N, "yref", terminal_state_ref)

        # Solve the MPC problem
        status = self.solver.solve()
        #if status != 0:
        #    raise ValueError(f"Acados solver failed with status {status}")

        # Extract the first control action
        u_optimal = self.solver.get(0, "u")

        # Extract the predictions
        x_pred = np.zeros((self.N + 1, self.dim_states))
        u_pred = np.zeros((self.N, self.dim_inputs))
        for i in range(self.N + 1):
            x_pred[i, :] = self.solver.get(i, "x")
            if i < self.N:
                u_pred[i, :] = self.solver.get(i, "u")

        if self.verbose:
            print(f"Optimal control action: {u_optimal}")
            print(f"x_pred: {x_pred}")
            print(f"u_pred: {u_pred}")

        return u_optimal, x_pred, u_pred
    






'''------Robust Model Predictive Control (RMPC)------'''

# Derived class for robust MPC Controller
#   Note: 1) assumes linearized dynamics, 2-norm cost, closed-loop control, stabilization task
#         2) implementation based on CasADi symbolic system and Acados
class LinearRMPCController(LQRController):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Qf: np.ndarray, 
            freq: float, 
            N: int, 
            K_feedback: Optional[np.ndarray] = None,  # Feedback gain for tube
            disturbance_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (lbz, ubz) of D
            max_iter: int = 10,  # Max iterations for invariant set computation
            name: str = 'RMPC', 
            type: str = 'RMPC', 
            verbose: bool = True
        ) -> None:

        self.Qf = Qf

        self.N = N  # Prediction horizon

        self.ocp = None
        self.solver = None

        super().__init__(env, dynamics, Q, R, freq, name, type, verbose)

        x0 = np.zeros(self.dynamics.dim_states)
        u0 = np.zeros(self.dynamics.dim_inputs)
        self.A, self.B = self.dynamics.get_linearized_AB_discrete(x0, u0, self.dt)

        # Automatically solve DARE if no K provided
        if K_feedback is None:
            from scipy.linalg import solve_discrete_are
            P = solve_discrete_are(self.A, self.B, Q, R)
            self.K_feedback = -np.linalg.inv(R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        else:
            self.K_feedback = K_feedback

        # Compute Omega_tube from disturbance box D 
        if disturbance_bounds is not None:

            disturbance_lbs = disturbance_bounds[0]
            disturbance_ubs = disturbance_bounds[1]

        elif self.env.disturbance_lbs is not None and self.env.disturbance_ubs is not None:

            disturbance_lbs = self.env.disturbance_lbs
            disturbance_ubs = self.env.disturbance_ubs
            
        else:
            raise ValueError("No bounds of additive disturbances provided, can not initialize RMPC")
        
        disturbance_lbs /= freq
        disturbance_ubs /= freq
        
        self.Omega_tube = self.compute_invariant_tube(self.A + self.B @ self.K_feedback, disturbance_lbs, disturbance_ubs, max_iter=max_iter)

        # Create polytope for tighten state constraints
        dim = len(self.env.state_lbs)
        H_box = np.vstack([np.eye(dim), -np.eye(dim)])
        h_box = np.hstack([self.env.state_ubs, -self.env.state_lbs])
        self.X = pt.Polytope(H_box, h_box)
        self.X_tighten = self.X - self.Omega_tube

        # Create polytope for tighten input constraints
        u_lbs_tighten = self.env.input_lbs + np.max(self.affine_map(self.Omega_tube, self.K_feedback).V, axis=0)
        u_ubs_tighten = self.env.input_ubs + np.min(self.affine_map(self.Omega_tube, self.K_feedback).V, axis=0)
        self.U_tighten = np.array([u_lbs_tighten, u_ubs_tighten])

        self.tube_bounds_x = self.estimate_bounds_from_polytope(self.Omega_tube)
        self.tube_bounds_u = np.abs(self.K_feedback @ self.tube_bounds_x)

        if self.verbose:
            print(f"Tighten state set X-Ω: {self.X_tighten.V}")
            print(f"Tube size x: {self.tube_bounds_x}")
            print(f"Tube size u: {self.tube_bounds_u}")

        self.setup()
    
    def affine_map(
        self, 
        poly: pt.Polytope, 
        A: np.ndarray
    ) -> pt.Polytope:
        
        """Compute the affine image of a polytope under x ↦ A x"""

        assert poly.V is not None, "No poly.V in Polytope! Can not apply affine map."

        V = poly.V  # vertices
        V_new = (A @ V.T).T

        return pt.Polytope(V_new)
        
    def compute_invariant_tube(
        self, 
        A_cl: np.ndarray, 
        lbz: np.ndarray, 
        ubz: np.ndarray, 
        tol: float = 1e-4, 
        max_iter: int = 10
    ) -> pt.Polytope:
        
        """
        Compute the robust positive invariant set Omega_tube using Minkowski recursion.

        This implementation uses the `polytope` library's Minkowski sum and affine map.
        """

        # Step 1: Define initial disturbance set D (box)
        dim = len(lbz)

        # H-representation: H x ≤ h
        H_box = np.vstack([np.eye(dim), -np.eye(dim)])
        h_box = np.hstack([ubz, -lbz])

        # Create polytope with both H-rep and V-rep
        D = pt.Polytope(H_box, h_box)

        # Step 2: Initialize Omega := D
        Omega = D

        for i in range(max_iter):
            # Step 3: Apply affine map A_cl to Omega: A_cl * Omega
            A_Omega = self.affine_map(Omega, A_cl)

            # Step 4: Minkowski sum: Omega_next = A_Omega ⊕ D
            Omega_next = A_Omega + D

            # Step 5: Check convergence via bounding box approximation
            bounds_old = self.estimate_bounds_from_polytope(Omega)
            bounds_new = self.estimate_bounds_from_polytope(Omega_next)

            if np.allclose(bounds_old, bounds_new, atol=tol):
                return Omega_next  # Return as vertices

            Omega = Omega_next

        return Omega  # Max iteration reached, return current estimate

    def estimate_bounds_from_polytope(
        self, 
        poly: pt.Polytope
    ) -> np.ndarray:
        
        """Estimate box bounds from polytope vertices (axis-aligned)."""
        vertices = poly.V
        return np.max(np.abs(vertices), axis=0)

    def setup(self) -> None:

        ## Model
        # Set up Acados model
        model = AcadosModel()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model.name = f"{self.name}_{timestamp}"

        # Define model: x_dot = f(x, u)
        model.x = self.dynamics.states
        model.u = self.dynamics.inputs
        model.f_expl_expr = ca.vertcat(self.dynamics.dynamics_function(self.dynamics.states, self.dynamics.inputs))
        model.f_impl_expr = None # no needed, we already have the explicit model

        ## Optimal control problem
        # Set up Acados OCP
        ocp = AcadosOcp()
        ocp.model = model # link to the model (class: AcadosModel)
        ocp.dims.N = self.N  # prediction horizon
        ocp.solver_options.tf = self.N * self.dt  # total prediction time
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" # Partially condensing interior-point method
        ocp.solver_options.integrator_type = "ERK" # explicit Runge-Kutta
        ocp.solver_options.nlp_solver_type = "SQP" # sequential quadratic programming

        # Set up cost function
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = np.block([
            [self.Q, np.zeros((self.dim_states, self.dim_inputs))],
            [np.zeros((self.dim_inputs, self.dim_states)), self.R],
        ])
        ocp.cost.W_e = self.Qf

        # Set up mapping from QP to OCP
        # Define output matrix for non-terminal state
        ocp.cost.Vx = np.block([
            [np.eye(self.dim_states)],
            [np.zeros((self.dim_inputs, self.dim_states))]
        ])
        # Define breakthrough matrix for non-terminal state
        ocp.cost.Vu = np.block([
            [np.zeros((self.dim_states, self.dim_inputs))],
            [np.eye(self.dim_inputs)]
        ])
        # Define output matrix for terminal state
        ocp.cost.Vx_e = np.eye(self.dim_states)

        # Initialize reference of task (stabilization)
        ocp.cost.yref = np.concatenate((self.target_state, np.zeros(self.dim_inputs)))
        ocp.cost.yref_e = self.target_state

        # Input constraints
        ocp.constraints.idxbu = np.arange(self.dim_inputs)

        if self.env.input_lbs is None:
            ocp.constraints.lbu = np.full(self.dim_inputs, -1e6)
        else:
            ocp.constraints.lbu = self.U_tighten[0]

        if self.env.input_ubs is None:
            ocp.constraints.ubu = np.full(self.dim_inputs, 1e6)
        else:
            ocp.constraints.ubu = self.U_tighten[1]

        # Expand initial state constraints (not here, do online)
        # Add Omega constraints on initial state x0: A x0 <= b
        ocp.dims.nh_0 = self.Omega_tube.A.shape[0]
        ocp.model.con_h_expr_0 = ca.mtimes(self.Omega_tube.A, ocp.model.x)
        ocp.constraints.lh_0 = -1e6 * np.ones(self.Omega_tube.A.shape[0])
        ocp.constraints.uh_0 = 1e6 * np.ones(self.Omega_tube.A.shape[0])  # placeholder

        # Expand tighten state constraints 
        ocp.dims.nh = self.X_tighten.A.shape[0]
        ocp.dims.nh_e = self.X_tighten.A.shape[0]
        ocp.model.con_h_expr = ca.mtimes(self.X_tighten.A, ocp.model.x)
        ocp.model.con_h_expr_e = ca.mtimes(self.X_tighten.A, ocp.model.x)
        ocp.constraints.lh = -1e6 * np.ones(self.X_tighten.A.shape[0])
        ocp.constraints.lh_e = -1e6 * np.ones(self.X_tighten.A.shape[0])
        ocp.constraints.uh = self.X_tighten.b.flatten()
        ocp.constraints.uh_e = self.X_tighten.b.flatten()

        # Recreate solver with tightened constraints
        self.ocp = ocp
        self.solver = AcadosOcpSolver(self.ocp, json_file=f"{self.name}.json", generate=True)
        
        if self.verbose:
            print("Tube-based MPC setup with constraint tightening completed.")

    @check_input_constraints
    def compute_action(
        self, 
        current_state: np.ndarray, 
        current_time: float
    ) -> np.ndarray:

        # Set upper limit of convex set equality constraint on target step to be 0
        lh_dynamic = self.Omega_tube.A @ current_state + self.Omega_tube.b.flatten()
        self.solver.constraints_set(0, "uh", lh_dynamic)

        status = self.solver.solve()

        x_nominal = self.solver.get(0, "x")
        u_nominal = self.solver.get(0, "u")

        # Apply tube feedback control
        u_real = u_nominal + self.K_feedback @ (current_state - x_nominal)

        if self.verbose:
            print("Current state:", current_state)
            print("Nominal state:", x_nominal)
            print("Nominal input:", u_nominal)
            print("Tube-corrected input:", u_real)

        # Also return nominal predictions
        x_pred = np.zeros((self.N + 1, self.dim_states))
        u_pred = np.zeros((self.N, self.dim_inputs))
        for i in range(self.N + 1):
            x_pred[i, :] = self.solver.get(i, "x")
            if i < self.N:
                u_pred[i, :] = self.solver.get(i, "u")

        return u_real, x_pred, u_pred, u_nominal
    
    def plot_robust_invariant_set(
        self, 
        lbz: np.ndarray, 
        ubz: np.ndarray, 
        tol: float = 1e-4, 
        max_iter: int = 10
    ) -> None:
        
        """
        Plot the 2D invariant set (Ω). Assumes 2D state space.
        """

        # Plot
        plt.figure(figsize=(6, 6))

        def plot_polytope(Omega: pt.Polytope, label=None):
            Omega_vertices = Omega.V
            assert Omega_vertices.shape[1] == 2, "Only 2D invariant sets are supported."

            # Convex hull of the Omega polytope
            hull = ConvexHull(Omega_vertices)
            hull_pts = Omega_vertices[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])

            plt.fill(hull_pts[:, 0], hull_pts[:, 1], color='red', alpha=0.1, label=label)
            plt.plot(hull_pts[:, 0], hull_pts[:, 1], color='red', linewidth=2)

        # Step 1: Define initial disturbance set D (box)
        dim = len(ubz)

        # H-representation: H x ≤ h
        H_box = np.vstack([np.eye(dim), -np.eye(dim)])
        h_box = np.hstack([ubz, -lbz])

        # Create polytope with both H-rep and V-rep
        D = pt.Polytope(H_box, h_box)

        # Step 2: Initialize Omega := D
        Omega = D
        plot_polytope(Omega, label = "Ω")

        A_cl = self.A + self.B @ self.K_feedback

        for i in range(max_iter):
            # Step 3: Apply affine map A_cl to Omega: A_cl * Omega
            A_Omega = self.affine_map(Omega, A_cl)

            # Step 4: Minkowski sum: Omega_next = A_Omega ⊕ D
            Omega_next = A_Omega + D

            # Step 5: Check convergence via bounding box approximation
            bounds_old = self.estimate_bounds_from_polytope(Omega)
            bounds_new = self.estimate_bounds_from_polytope(Omega_next)

            if np.allclose(bounds_old, bounds_new, atol=tol):
                return Omega_next  # Return as vertices

            Omega = Omega_next
            plot_polytope(Omega)

        plt.title("Robust Invariant Set Ω")
        plt.xlabel("Position p")
        plt.ylabel("Velocity v")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_tighten_state_set(self):
       
        X_tighten_vertices = self.X_tighten.V

        assert X_tighten_vertices.shape[1] == 2, "Only 2D tighten sets are supported."

        # Convex hull of the Omega polytope
        hull = ConvexHull(X_tighten_vertices)
        hull_pts = X_tighten_vertices[hull.vertices]

        # Plot the original state constraints
        X = np.array([
            [self.env.state_lbs[0], self.env.state_lbs[1]],
            [self.env.state_lbs[0], self.env.state_ubs[1]],
            [self.env.state_ubs[0], self.env.state_ubs[1]],
            [self.env.state_ubs[0], self.env.state_lbs[1]],
            [self.env.state_lbs[0], self.env.state_lbs[1]]
        ])

        # Plot
        plt.figure(figsize=(6, 6))
        plt.fill(hull_pts[:, 0], hull_pts[:, 1], color='skyblue', alpha=0.5, label='X-Ω')
        plt.plot(hull_pts[:, 0], hull_pts[:, 1], 'b-', linewidth=2)
        plt.plot(X[:, 0], X[:, 1], 'r--', linewidth=2, label='X')

        plt.title("Tighten state Set X-Ω and Original set X")
        plt.xlabel("State x₁")
        plt.ylabel("State x₂")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()



