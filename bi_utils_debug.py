import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from config import dt, N, u_a_max, u_a_min, u_beta_max, u_beta_min, T_sim, total_steps_sim, rear_dist, r1, r2
import casadi as ca
import torch
import random

class BicycleDynamics(torch.nn.Module):                         # pytorch 版本的dynamics，训练用
    def __init__(self):
        super(BicycleDynamics, self).__init__()

    def forward(self, t, z_and_u):
        """
        Args (Bicycle):
            t: time (required by torchdiffeq, even if not used)
            z_and_u: concatenated state [x, y, theta, speed] and control [a, beta] (torch tensor)
            rear_dist: distance from rear axle to center of gravity (constant parameter)

        Returns:
            dx/dt (torch tensor)
        """
        # Split state and control from concatenated tensor
        z = z_and_u[0,:4]  # state [x, y, theta, speed]
        u = z_and_u[0,4:]  # control input [a, beta]

        speed = z[3]
        theta = z[2]
        u_a = u[0]
        u_beta = u[1]

        # Compute derivatives
        x_dot = speed * torch.cos(theta) - speed * torch.sin(theta) * u_beta
        y_dot = speed * torch.sin(theta) + speed * torch.cos(theta) * u_beta
        theta_dot = speed/rear_dist * u_beta
        speed_dot = u_a
        dzdt = torch.stack([x_dot, y_dot, theta_dot, speed_dot, \
                            torch.tensor(0.0, device=z.device), \
                            torch.tensor(0.0, device=z.device)])    # 这里的dxdt 是一个 6 维的 tensor，前面四维是状态的导数，后面两维是控制输入的导数（因为控制输入在这个模型里是直接给定的，所以它们的导数是0）
        dzdt = dzdt.unsqueeze(0)
        return dzdt

def plot_traj(state_mpc, u_mpc, time, h, option):
    import os
    os.makedirs("./figs", exist_ok=True)
    x_mpc = state_mpc[:, 0]
    y_mpc = state_mpc[:, 1]
    theta_mpc = state_mpc[:,2]

    v_mpc = u_mpc[:,0]
    w_mpc = u_mpc[:,1]

    # Plot State Trajectory
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    len_state_mpc = len(x_mpc)
    len_u_mpc = len(v_mpc)
    plt.plot(time[:len_state_mpc], x_mpc, linestyle='-', dashes=[3, 1], linewidth=5, label=r"$x_{mpc}$")
    plt.plot(time[:len_state_mpc], y_mpc, linestyle='-', dashes=[3, 1], linewidth=5, label=r"$y_{mpc}$")
    plt.plot(time[:len_state_mpc], theta_mpc, linestyle='-', dashes=[3, 1], linewidth=5, label=r"$\theta_{mpc}$")
    plt.xlabel("Time (s)",fontsize=20, fontweight='bold')
    plt.ylabel("State Trajectory",fontsize=20, fontweight='bold')
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.xticks(fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')

    # Plot Control input trajectory
    plt.subplot(1, 2, 2)

    plt.plot(time[:len_u_mpc], v_mpc, linestyle='-', dashes=[3, 1], linewidth=5, label=r"$v_{mpc}$")
    plt.plot(time[:len_u_mpc], w_mpc, linestyle='-', dashes=[3, 1], linewidth=5, label=r"$w_{mpc}$")
    plt.xlabel("Time (s)",fontsize=20, fontweight='bold')
    plt.ylabel("Control Input",fontsize=20, fontweight='bold')
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.xticks(fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')
    plt.tight_layout()
    output_dir = f'./figs/mpc_N{N}_h{h}_{option}.png'
    plt.savefig(output_dir, dpi=300)
    plt.close()

    print(f"Figure saved to {output_dir}")
    
def bicycle_dynamics(z, u):                                  # Numpy 版本的dynamics，仿真用
    theta, speed = z[0,2], z[0,3]
    u_a, u_beta = u[0], u[1]
    x_dot = speed * np.cos(theta) - speed * np.sin(theta) * u_beta
    y_dot = speed * np.sin(theta) + speed * np.cos(theta) * u_beta
    theta_dot = speed/rear_dist * u_beta
    speed_dot = u_a
    return np.array([x_dot, y_dot, theta_dot, speed_dot])

def rk4(z, u):
    # RK4 integration step for dynamics
    k1 = bicycle_dynamics(z, u)
    k2 = bicycle_dynamics(z + dt / 2 * k1, u)
    k3 = bicycle_dynamics(z + dt / 2 * k2, u)
    k4 = bicycle_dynamics(z + dt * k3, u)
    z_next = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)   # 这里simulation的更新使用rk4，但是训练的时候是用torchdiffeq的odeint来更新的，训练和仿真使用不同的数值积分方法，主要是为了让训练更快一些，因为torchdiffeq的odeint在训练过程中可以自动计算梯度，而rk4需要手动实现反向传播，这样会比较麻烦，所以在训练的时候我们选择使用torchdiffeq的odeint来更新状态。
    return z_next

def bicycle_solve_qp(lambda_x, lambda_y, lambda_theta, lambda_speed, theta, speed):     # 这里的公式和train.py是不同的，因为train.py是在无约束条件下计算控制律，而QP用于求解有约束的情况

    # Define decision variables
    u_a = ca.SX.sym('u_a')                                   # 数学写法是 argmin_v H(v, omega)，这里的v和omega是优化变量，所以用ca.SX.sym来定义符号变量
    u_beta = ca.SX.sym('u_beta')

    # Define the Hamiltonian
    H = (r1*u_a**2 + r2*u_beta**2 -
        lambda_x * u_beta * speed * ca.sin(theta) +
        lambda_y * u_beta * speed * ca.cos(theta) +
        lambda_theta * u_beta * speed/rear_dist   +
        lambda_speed * u_a)

    # Set up the QP problem
    qp = {
        'x': ca.vertcat(u_a, u_beta),  # Decision variables [u_a, u_beta]
        'f': H,                    # Cost function (Hamiltonian)
        'g': ca.vertcat()          # No additional equality/inequality constraints
    }

    # Set bounds for the decision variables

    lbx = [u_a_min, u_beta_min]  # Lower bounds for [u_a, u_beta]
    ubx = [u_a_max, u_beta_max]  # Upper bounds for [u_a, u_beta]

    opts = {
    'printLevel': 'none'  # Suppress solver output for qpoases
    }
    # Create QP solver
    S = ca.qpsol('S', 'qpoases', qp, opts)

    # Solve the problem
    solution = S(lbx=lbx, ubx=ubx)

    # Extract results
    u_a_opt = solution['x'][0]
    u_beta_opt = solution['x'][1]
    return u_a_opt, u_beta_opt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_animation(t_span, x_traj, y_traj, theta_traj, costate_trajectory, option):

    # Set up figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Left plot: State trajectory
    axs[0].set_xlabel("Time (s)",fontsize=20, fontweight='bold')
    axs[0].set_ylabel("State Trajectory",fontsize=20, fontweight='bold')
    line_state1, = axs[0].plot([], [], linestyle='-', dashes=[3, 1], label=r"$x_{ncr}$", linewidth=5)
    line_state2, = axs[0].plot([], [], linestyle='-', dashes=[3, 1], label=r"$y_{ncr}$", linewidth=5)
    line_state3, = axs[0].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\theta_{ncr}$", linewidth=5)
    axs[0].legend(fontsize=28)
    axs[0].grid(True)
    axs[0].set_xlim(0, T_sim)
    axs[0].set_ylim(-5, 5)
    axs[0].tick_params(axis="both", labelsize=20)

    # Right plot: Co-state trajectory
    axs[1].set_xlabel("Time (s)", fontsize=20, fontweight='bold')
    axs[1].set_ylabel("Lambda values", fontsize=20, fontweight='bold')
    # Initialize co-state lines
    line_lambda1, = axs[1].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\lambda{x}$", linewidth=5)
    line_lambda2, = axs[1].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\lambda{y}$", linewidth=5)
    line_lambda3, = axs[1].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\lambda{\theta}$", linewidth=5)
    axs[1].legend(fontsize=28)
    axs[1].grid(True)
    axs[1].set_ylim(-10, 10)
    axs[1].tick_params(axis="both", labelsize=20)

    
    # Animation update function
    def update(frame):
        # Update state trajectory plot
        line_state1.set_data(t_span[:frame+1], x_traj[:frame+1])
        line_state2.set_data(t_span[:frame+1], y_traj[:frame+1])
        line_state3.set_data(t_span[:frame+1], theta_traj[:frame+1])

        # # Update co-state trajectory plot
        n = len( costate_trajectory[frame][0, :, 0])
        t_span_costate = np.linspace(frame*dt, (frame+n-1)*dt, n)
        axs[1].set_xlim(frame*dt, (frame+n-1)*dt)
        line_lambda1.set_data(t_span_costate, costate_trajectory[frame][0, :, 0])
        line_lambda2.set_data(t_span_costate, costate_trajectory[frame][0, :, 1])
        line_lambda3.set_data(t_span_costate, costate_trajectory[frame][0, :, 2])

        return line_state1, line_state2, line_state3, line_lambda1, line_lambda2, line_lambda3

    # Create animation
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=total_steps_sim, interval=100, blit=True)
    output_dir = f"./figs/animation_{option}.gif"
    ani.save(output_dir, writer=animation.PillowWriter(fps=20))
    print(f"Animation saved to {output_dir}")