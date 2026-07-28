from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from config import (dt, T_sim, total_steps_sim, n, 
                    u_a_min, u_a_max, u_s_min, u_s_max,
                    rear_dist, h1, h2, h3, h4,
                    q1,q2,q3,q4,r1,r2,
                    x_ref,y_ref,theta_ref,speed_ref,
                    state_0a,state_0b)
from bi_utils_debug import bicycle_solve_qp, rk4
from matplotlib.patches import Rectangle
import casadi as ca
from concurrent.futures import ProcessPoolExecutor

x_lw_bound = 0
y_lw_bound = 0
theta_lw_bound = 0
x_up_bound = 6
y_up_bound = 6
theta_up_bound = 3.14
N_init_states = 100
hit = 0
x_ref = 0
y_ref = 0
theta_ref = 0
speed_ref = 0
converge_thre = 0.4

prediction_time = n * dt
# dt_mpc = 0.1
# dt_pred = 0.1
dtmpc_ratio = 1 #int(dt_mpc/dt)
dtpred_ratio = 1 # int(dt_pred/dt)
dt_mpc = dtmpc_ratio * dt
dt_pred = dtpred_ratio * dt
mpc_pred_horizon = int(prediction_time / dt_pred)
print(f'dt = {dt}\n'
      f'dt_mpc = {dt_mpc}\n'
      f'dt_pred = {dt_pred}\n'
      f'mpc prediction horizon = {mpc_pred_horizon}')

seed = 1
np.random.seed(seed)

class bi_MPC:
    def __init__(self, dt, dt_pred, n, mpc_pred_horizon, u_a_min, u_a_max, u_s_min, u_s_max):
        # System parameters
        self.dt = dt
        self.dt_pred = dt_pred
        # self.n = n
        self.mpc_pred_horizon = mpc_pred_horizon
        self.u_a_min = u_a_min
        self.u_a_max = u_a_max
        self.u_s_min = u_s_min
        self.u_s_max = u_s_max

        self._build_solver()
    
    def _build_solver(self):
        HN = self.mpc_pred_horizon

        U = ca.MX.sym("U", 2, HN)
        P = ca.MX.sym("P", 4 + 4 * (HN + 1), 1)

        z_k = P[0:4]
        ref_traj = ca.reshape(P[4:], 4, HN + 1)

        Q = np.diag([q1, q2, q3, q4])
        R = np.diag([r1, r2])
        Hmat = np.diag([h1, h2, h3, h4])

        obj = 0
        for k in range(HN):
            u_k = U[:, k]
            ref_k = ref_traj[:, k]

            obj += ca.mtimes((z_k - ref_k).T, Q @ (z_k - ref_k)) * dt_pred
            obj += ca.mtimes(u_k.T, R @ u_k) * dt_pred

            z_k = self.rk4(z_k, u_k, dt_pred)

        ref_terminal = ref_traj[:, HN]
        obj += ca.mtimes((z_k - ref_terminal).T, Hmat @ (z_k - ref_terminal))

        U_flat = ca.reshape(U, -1, 1)

        nlp = {
            "f": obj,
            "x": U_flat,
            "p": P,
        }

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
        }

        self.solver = ca.nlpsol("S", "ipopt", nlp, opts)
        self.lbu = [self.u_a_min, self.u_s_min] * HN
        self.ubu = [self.u_a_max, self.u_s_max] * HN
        

    def dynamics(self, z, u):
        # Bicycle model dynamics
        theta = z[2]
        speed = z[3]
        u_a, u_s = u[0], u[1]
        x_dot = speed * ca.cos(theta) - speed * ca.sin(theta) * u_s
        y_dot = speed * ca.sin(theta) + speed * ca.cos(theta) * u_s
        theta_dot = speed/rear_dist * u_s
        speed_dot = u_a
        return ca.vertcat(x_dot, y_dot, theta_dot, speed_dot)

    def rk4(self, z, u, dt_step):    # for prediction use dt_pred rather than dt
        # RK4 integration step for dynamics
        k1 = self.dynamics(z, u)
        k2 = self.dynamics(z + dt_step / 2 * k1, u)
        k3 = self.dynamics(z + dt_step / 2 * k2, u)
        k4 = self.dynamics(z + dt_step * k3, u)
        z_next = z + dt_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return z_next

    def get_control_input(self, current_state, ref_traj, initial_input_guess):
        current_state = np.asarray(current_state).reshape(-1)
        ref_traj = np.asarray(ref_traj)

        p_value = np.concatenate([
            current_state,
            ref_traj.reshape(-1, order="F"),
        ]).reshape(-1, 1)

        sol = self.solver(
            x0=initial_input_guess,
            lbx=self.lbu,
            ubx=self.ubu,
            p=p_value,
        )

        return sol["x"].full()
    
def single_mpc_sim(state_0):
    controller = bi_MPC(dt, dt_pred, n, mpc_pred_horizon, u_a_min, u_a_max, u_s_min, u_s_max)
    # state_0 = np.array([x0, y0, theta0, speed0])
    # state_0 = all_states[j][:]
    state_traj_undisturbed = [state_0]
    x_k = state_0
    control_traj = []
    # Simulation loop
    u_k = np.array([0,0])
    u_pred_traj = np.zeros((2 * mpc_pred_horizon,1))
    # simulation
    for k in range(total_steps_sim):
        # Calculate control input
        # ref_traj_segment = traj_ref[:, k:k + n + 1]
        traj_ref = np.zeros((4, total_steps_sim + n))
        ref_traj_segment = traj_ref[:, k : k + mpc_pred_horizon * dtpred_ratio + 1 : dtpred_ratio]
        if (k) % dtmpc_ratio == 0: 
            u_pred_traj = controller.get_control_input( x_k, ref_traj_segment, 
                initial_input_guess = np.vstack([u_pred_traj[2:],u_pred_traj[-2:]])
                )
            u_k = u_pred_traj[:2]
            # print(f'MPC input update')
            # print(f'MPC N={n} - timestep: {k} finished')
        
        control_traj.append(u_k)
        x_k = controller.rk4(x_k, u_k, dt)
        x_k = x_k.full().flatten().tolist()
        state_traj_undisturbed.append(x_k)

    # plot the current trajectory
    state_traj_undisturbed = np.array(state_traj_undisturbed)
    x_traj = state_traj_undisturbed[:,0]
    y_traj = state_traj_undisturbed[:,1]
    theta_traj = state_traj_undisturbed[:,2]
    speed_traj = state_traj_undisturbed[:,3]
    abs_convergence_err = abs(x_traj[-1] - x_ref) + abs(y_traj[-1] - y_ref) + abs(theta_traj[-1] - theta_ref) + abs(speed_traj[-1] - speed_ref)
    good_converge = False
    if abs_convergence_err <= converge_thre:
        good_converge = True
    #     hit = hit+1
    # plot_bicycle_traj(ax=ax,x_robot=x_traj,y_robot=y_traj,x_ref=x_ref,y_ref=y_ref, good_converge=good_converge)
    # print(f"state_0_{j} = np.array([[{x0:.2f}, {y0:.2f}, {theta0:.2f}, {speed0:.2f}]]) Convergence: {good_converge}")

    return {
        'x_traj': x_traj,
        'y_traj': y_traj,
        'converged': good_converge,
        'error': abs_convergence_err,
        'state_0': state_0
    }
    
def plot_bicycle_traj(ax, x_robot, y_robot,x_ref,y_ref, good_converge):
    color = 'tab:red'
    if good_converge:
        color = 'tab:green'
    x_robot = np.asarray(x_robot).reshape(-1)
    y_robot = np.asarray(y_robot).reshape(-1)
    if len(x_robot)!=len(y_robot):
        raise ValueError("Lengths of x and y don't match.")
    if len(x_robot)==0:
        raise ValueError("Zero length for x and y")
    # start_xy = np.array([x_robot[0],y_robot[0]])
    # goal_xy = np.array([x_ref,y_ref])
    line, = ax.plot(x_robot, y_robot, color=color, linewidth=0.5, alpha=0.6)
    ax.scatter(x_robot[0], y_robot[0], color=color, s=5, marker="o", alpha=1, zorder=5)

import time

if __name__ == '__main__':
    start_time = time.time()
    # initial states
    all_states = []
    for j in range(N_init_states):
        # x0 = np.random.uniform(x_lw_bound, x_up_bound) * np.random.choice([-1,1])
        # y0 = np.random.uniform(y_lw_bound, y_up_bound) * np.random.choice([-1,1])
        x0 = np.random.uniform(-x_up_bound,x_up_bound)
        y0 = np.random.uniform(-y_up_bound,y_up_bound)
        while (abs(x0)<x_lw_bound) and (abs(y0)<y_lw_bound):
            x0 = np.random.uniform(-x_up_bound,x_up_bound)
            y0 = np.random.uniform(-y_up_bound,y_up_bound)
        theta0 = np.random.uniform(theta_lw_bound, theta_up_bound) * np.random.choice([-1,1])
        speed0 = 0
        all_states.append([x0,y0,theta0,speed0])

    # parallel execution setup
    num_cores = 20
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(single_mpc_sim, all_states)
    results = list(results)

    # 
    fig, ax = plt.subplots(figsize=(8,6))
    hit = 0

    for result in results:
        if result['converged']:
            hit += 1
        plot_bicycle_traj(ax=ax,x_robot=result['x_traj'],y_robot=result['y_traj'],x_ref=x_ref,y_ref=y_ref, good_converge=result['converged'])
        print(f"state_0 = {result['state_0']} Convergence: {result['converged']}")

    print(f"Coverged rate: {hit/N_init_states:.2f}")

    outer_box = Rectangle((-x_up_bound, -y_up_bound),2 * x_up_bound,2 * y_up_bound,
        fill=False,
        linestyle=":",
        linewidth=1.5,
        edgecolor="black",
    )
    inner_box = Rectangle((-x_lw_bound, -y_lw_bound),2 * x_lw_bound,2 * y_lw_bound,
        fill=False,
        linestyle=":",
        linewidth=1.5,
        edgecolor="black",
    )
    ax.add_patch(inner_box)
    ax.add_patch(outer_box)
    ax.scatter(x_ref, y_ref, color="black", s=10, label="Goal", zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    output_dir = f"./bi_figs/MPC_rate{hit/N_init_states:.2f}_N{N_init_states}_dt{dt}_Tsim{T_sim}_dtmpc{dt_mpc}_dtpred{dt_pred}_seed{seed}_thre{converge_thre}_{x_lw_bound}-{x_up_bound}-{y_lw_bound}-{y_up_bound}.png"
    plt.savefig(output_dir, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {output_dir}")

    print(f"Executed in {(-start_time+time.time()):.2f}")
