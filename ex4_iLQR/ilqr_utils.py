import numpy as np


from utils.env import Env, Dynamics
from utils.controller import check_input_constraints
from ex2_LQR.lqr_utils import LQRController





# Derived class for iLQR Controller
class iLQRController(LQRController):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Qf: np.ndarray, 
            freq: float, 
            name: str = 'iLQR', 
            type: str = 'iLQR', 
            max_iter: int = 100, 
            tol: float = 1e-1, 
            verbose: bool = True
        ) -> None:

        self.Qf = Qf  # Terminal cost matrix

        self.max_iter = max_iter
        self.tol = tol

        self.K_k_arr = None
        self.u_kff_arr = None
        self.total_cost_list = []  # Store total cost per iteration

        super().__init__(env, dynamics, Q, R, freq, name, type, verbose)

    def setup(self, input_traj: np.ndarray) -> None:
        """
        Perform iLQR to compute the optimal control sequence.
        """
        self.x_eq = self.target_state
        self.u_eq = self.dynamics.get_equilibrium_input(self.x_eq)

        N = len(input_traj)

        x_traj = np.zeros((self.dim_states, N+1))
        u_traj = np.copy(input_traj)
        x_traj[:, 0] = self.init_state

        self.K_k_arr = np.zeros((self.dim_states, N))
        self.u_kff_arr = np.zeros((N,))

        for n in range(self.max_iter):
            for k in range(N):
                next_state = self.dynamics.one_step_forward(current_state=x_traj[:, k], current_input=u_traj[k], dt=self.dt)
                x_traj[:, k + 1] = next_state

            x_N_det = x_traj[:, -1] - self.target_state
            x_N_det = x_N_det.reshape(-1, 1)

            s_k_bar = (x_N_det.T @ self.Qf @ x_N_det) / 2
            s_k = self.Qf @ x_N_det
            S_k = self.Qf

            for k in range(N - 1, -1, -1):
                A_lin, B_lin = self.dynamics.get_linearized_AB_discrete(current_state=x_traj[:, k], current_input=u_traj[k], dt=self.dt)

                x_k_det = x_traj[:, k] - self.target_state
                x_k_det = x_k_det.reshape(-1, 1)
                
                g_k_bar = (x_k_det.T @ self.Q @ x_k_det + self.R * u_traj[k] ** 2) * self.dt / 2
                q_k = (self.Q @ x_k_det) * self.dt
                Q_k = (self.Q) * self.dt
                r_k = (self.R * u_traj[k]) * self.dt
                R_k = (self.R) * self.dt
                P_k = np.zeros((2,)) * self.dt

                l_k = (r_k + B_lin.T @ s_k)
                G_k = (P_k + B_lin.T @ S_k @ A_lin)
                H_k = (R_k + B_lin.T @ S_k @ B_lin)

                det_u_kff = - np.linalg.inv(H_k) @ l_k
                K_k = - np.linalg.inv(H_k) @ G_k
                u_kff = u_traj[k] + det_u_kff - (K_k @ x_traj[:, k])

                self.K_k_arr[:, k] = (K_k.T).flatten()
                self.u_kff_arr[k] = u_kff.item()

                s_k_bar = g_k_bar + s_k_bar + (det_u_kff.T @ H_k @ det_u_kff) / 2 + det_u_kff.T @ l_k
                s_k = q_k + A_lin.T @ s_k + K_k.T @ H_k @ det_u_kff + K_k.T @ l_k + G_k.T @ det_u_kff
                S_k = Q_k + A_lin.T @ S_k @ A_lin + K_k.T @ H_k @ K_k + K_k.T @ G_k + G_k.T @ K_k

                if self.verbose:
                    print(f"s_k_bar: {s_k_bar}")
                    print(f"s_k: {s_k}")
                    print(f"S_k: {S_k}")

                    print(f"A_lin: {A_lin}")
                    print(f"B_lin: {B_lin}")

                    print(f"x_k_det: {x_k_det}")

                    print(f"g_k_bar: {g_k_bar}")
                    print(f"q_k: {q_k}")
                    print(f"Q_k: {Q_k}")
                    print(f"r_k: {r_k}")
                    print(f"R_k: {R_k}")
                    print(f"P_k: {P_k}")

                    print(f"l_k: {l_k}")
                    print(f"G_k: {G_k}")
                    print(f"H_k: {H_k}")

                    print(f"det_u_kff: {det_u_kff}")
                    print(f"K_k: {K_k}")

                    print(f"A_lin.T @ S_k @ A_lin: {A_lin.T @ S_k @ A_lin}")
                    print(f"A_lin.T @ s_k: {A_lin.T @ s_k}")
                    print(f"K_k.T @ H_k @ K_k: {K_k.T @ H_k @ K_k}")
                    print(f"G_k.T @ K_k: {G_k.T @ K_k}")
                
            new_u_traj = np.zeros_like(u_traj)
            new_x_traj = np.zeros_like(x_traj)
            new_x_traj[:, 0] = self.init_state

            for k in range(N):
                new_u_traj[k] = self.u_kff_arr[k] + self.K_k_arr[:, k].T @ new_x_traj[:, k]
                next_state = self.dynamics.one_step_forward(current_state=new_x_traj[:, k], current_input=new_u_traj[k], dt=self.dt)
                new_x_traj[:, k + 1] = next_state

            # ---- Compute total cost for this iteration ----
            total_cost = 0.0
            for k in range(N):
                x_k_det = x_traj[:, k] - self.target_state
                total_cost += 0.5 * (x_k_det.T @ self.Q @ x_k_det + self.R * u_traj[k] ** 2)
            x_N_det = x_traj[:, -1] - self.target_state
            total_cost += 0.5 * (x_N_det.T @ self.Qf @ x_N_det)
            self.total_cost_list.append(total_cost.item())

            if np.max(np.abs(new_u_traj - u_traj)) < self.tol:
                print(f"Use {n} iteration until converge.")
                break
            else:
                print(f"Iteration {n}: residual error is {np.max(np.abs(new_u_traj - u_traj))}")

            u_traj = new_u_traj
            x_traj = new_x_traj

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_step: int) -> np.ndarray:
        if self.K_k_arr is None or self.u_kff_arr is None:
            raise ValueError("iLQR parameters are not computed. Call setup() first.")

        u = self.u_kff_arr[current_step] + self.K_k_arr[:, current_step].T @ current_state
        return u