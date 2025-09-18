import numpy as np
import sympy as sp
import pickle
from abc import ABC, abstractmethod
from typing import List, Callable, Union, Optional, Tuple, Any


from utils.env import Env, Dynamics
from utils.controller import BaseController, check_input_constraints





# Derived class for (symbolic) dynamic programming (DP) controller
# Note: 1) only applicable for linear system
#       2) return the symbolic control law, implementation based on sympy
class DPController(BaseController):
    def __init__(
            self,
            env: Env,
            dynamics: Dynamics,
            Q: np.ndarray,
            R: np.ndarray,
            Qf: np.ndarray, 
            freq: float,
            Horizon: int,
            name: str = 'DP',
            type: str = 'DP', 
            verbose: bool = False,
            bangbang: bool = False,
            symbolic_weight: bool = False
        ) -> None:

        """
        DP solver for linear system:
            x_{k+1} = A x_k + B u_k
            cost = sum u_k^T R u_k + terminal cost
        """

        super().__init__(env, dynamics, freq, name, verbose)

        self.N = Horizon

        self.Q = Q
        self.R = R
        self.Qf = Qf

        # Solve input at equilibrium
        self.u_eq = self.dynamics.get_equilibrium_input(self.init_state)

        # Get linearized & discretized A and B matrices
        self.Ad, self.Bd = self.dynamics.get_linearized_AB_discrete(self.init_state, 0, self.dt)

        self.bangbang = bangbang # True or False
        self.symbolic_weight = symbolic_weight # True or False

        # Define symbolic variables
        self.x_sym = None
        self.u_sym = None
        self.Q_sym = None
        self.R_sym = None
        self.Qf_sym = None
        self.x_ref_sym = None

        # Store results
        self.J_sym = [None] * (self.N + 1)
        self.mu_sym = [None] * self.N
    
        self.setup()

    def setup(self) -> None:

        # Create symbolic states and inputs
        self.x_sym = [sp.Matrix(sp.symbols(f'p_{k} v_{k}')) for k in range(self.N + 1)]
        self.u_sym = [sp.Symbol(f'u_{k}') for k in range(self.N)]
        # Create symbolic weight matrices
        if self.symbolic_weight:
            q1, q2 = sp.symbols('q_p q_v')
            self.Q_sym = sp.diag(q1, q2)
            self.R_sym = sp.Symbol('r')
            self.Qf_sym = sp.diag(q1, q2)
            # Create symbolic reference state
            p_ref, v_ref = sp.symbols('p_ref v_ref')
            x_ref = [p_ref, v_ref]
            self.x_ref_sym = sp.Matrix(x_ref) 
        else:
            self.Q_sym = sp.Matrix(self.Q)
            self.R_sym = sp.Float(self.R)
            self.Qf_sym = sp.Matrix(self.Qf)
            # Create numpy reference state
            self.x_ref_sym = sp.Matrix(self.target_state) 

        # Make a copy
        J, mu = self.J_sym, self.mu_sym

        # Terminal cost: J_N = (x_N - x_ref)^T Qf (x_N - x_ref)
        err_N = self.x_sym[self.N] - self.x_ref_sym
        J[self.N] = (err_N.T * self.Qf_sym * err_N)[0, 0]

        for k in reversed(range(self.N)):

            # x_{k+1} = A x_k + B u_k
            x_next = self.Ad * self.x_sym[k] + self.Bd * self.u_sym[k]

            # Cost at step k: u_k^T R u_k + J_{k+1}(x_{k+1})
            stage_cost = self.R_sym * self.u_sym[k]**2
            J_kplus1_sub = J[k + 1].subs({self.x_sym[k + 1][i]: x_next[i] for i in range(2)})

            total_cost = stage_cost + J_kplus1_sub

            # Compute the optimal control input and cost-to-go
            if self.bangbang: # bangbang input (MIP)
                mu_k, J_k = self.solve_bangbang(total_cost, k)
            else: # continious input (NLP)
                mu_k, J_k = self.solve_continuous(total_cost, k)
            
            # Store the symbolic expressions
            mu[k] = mu_k
            J[k] = J_k
        
        # Log the symbolic expressions back
        self.J_sym = J
        self.mu_sym = mu

        if self.verbose:
            print(f"Dynamic Programming policy with input constraints computed.")
            self.print_solution()

    def solve_continuous(
            self, 
            total_cost: sp.Expr, 
            k: int
        ) -> Tuple[sp.Expr, sp.Expr]:

        # Derivative w.r.t u_k
        dJ_du = sp.diff(total_cost, self.u_sym[k])
        u_star = sp.solve(dJ_du, self.u_sym[k])[0]
        mu_k = sp.simplify(u_star)

        # Plug u_k* back into cost to get J_k
        cost_k_opt = total_cost.subs(self.u_sym[k], u_star)
        J_k = sp.simplify(cost_k_opt)

        return mu_k, J_k
    
    def solve_bangbang(
            self, 
            total_cost: sp.Expr, 
            k: int
        ) -> Tuple[sp.Expr, sp.Expr]:

        # Evaluate cost for u_k = -1 and u_k = 1
        cost_minus1 = sp.simplify(total_cost.subs(self.u_sym[k], -1))
        cost_plus1  = sp.simplify(total_cost.subs(self.u_sym[k], 1))

        # Try subtracting to simplify the condition
        delta_cost = sp.simplify(cost_plus1 - cost_minus1)

        # Store optimal control policy as piecewise
        mu_k = sp.Piecewise(
            (-1, delta_cost > 0),
            (1,  True)  # fallback
        )
        # OR mu_k = sp.simplify(mu_k)
        # OR mu_k = sp.piecewise_fold(mu_k)

        # Store cost-to-go as piecewise
        J_k = sp.Piecewise(
            (cost_minus1, delta_cost > 0),
            (cost_plus1,  True)
        )
        # OR J_k = sp.simplify(J_k)
        # OR J_k = sp.piecewise_fold(J_k)

        return mu_k, J_k

    def print_solution(self):
        for k, uk in enumerate(self.mu_sym):
            print(f"u_{k}*(x_{k}) =", uk)
        #print("\nJ_0(x_0) =")
        #sp.pprint(self.J_sym[0])

    def save_policy(
            self, 
            filename: str = 'dp_policy.pkl'
        ) -> None:

        with open(filename, 'wb') as f:
            pickle.dump({
                'mu_sym': self.mu_sym,
                'J_sym': self.J_sym,
            }, f)

    def load_policy(
            self, 
            filename: str = 'dp_policy.pkl'
        ) -> None:

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.mu_sym = data['mu_sym']
            self.J_sym = data['J_sym']

    @check_input_constraints
    def compute_action(
        self, 
        current_state: np.ndarray, 
        current_step: int
    ) -> np.ndarray:

        if any(mu is None for mu in self.mu_sym):
            raise ValueError("DP Policy is not computed yet. Call setup() first.")
        
        if current_step >= self.N:
            raise ValueError(f"current_step = {current_step} exceeds or equals horizon N = {self.N}")
        
        mu_expr = self.mu_sym[current_step]

        # Substitute current state into the symbolic expression
        if self.symbolic_weight:
            subs_dict = {
                sp.Symbol(f'p_{current_step}'): current_state[0],
                sp.Symbol(f'v_{current_step}'): current_state[1],
                sp.Symbol('q_p'): self.Q[0, 0],
                sp.Symbol('q_v'): self.Q[1, 1],
                sp.Symbol('r'): self.R[0, 0],
                sp.Symbol('p_ref'): self.target_state[0], 
                sp.Symbol('v_ref'): self.target_state[1]
            }
        else:
            subs_dict = {
                sp.Symbol(f'p_{current_step}'): current_state[0],
                sp.Symbol(f'v_{current_step}'): current_state[1],
            }
    
        mu = float(mu_expr.subs(subs_dict).evalf())

        mu += self.u_eq  # Add equilibrium input

        if self.verbose:
            print(f"state: {current_state}, optimal input: {mu}")

        return mu
    