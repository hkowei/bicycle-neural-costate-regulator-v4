import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from config import dt, N, u_a_max, u_a_min, u_s_max, u_s_min, T_sim, total_steps_sim, rear_dist, tot_dist, r1, r2, bi_scaling
import casadi as ca
import torch
import random
import os
import contextlib

# class BicycleDynamics(torch.nn.Module):                         # pytorch 版本的dynamics，训练用
#     def __init__(self):
#         super(BicycleDynamics, self).__init__()

#     def forward(self, t, z_and_u):
#         """
#         Args (Bicycle):
#             t: time (required by torchdiffeq, even if not used)
#             z_and_u: concatenated state [x, y, theta, speed] and control [a, beta] (torch tensor)
#             rear_dist: distance from rear axle to center of gravity (constant parameter)

#         Returns:
#             dx/dt (torch tensor)
#         """
#         # Split state and control from concatenated tensor
#         z = z_and_u[0,:4]  # state [x, y, theta, speed]
#         u = z_and_u[0,4:]  # control input [a, beta]

#         speed = z[3]
#         theta = z[2]
#         u_a = u[0]
#         u_beta = u[1]

#         # Compute derivatives
#         x_dot = speed * torch.cos(theta) - speed * torch.sin(theta) * u_beta
#         y_dot = speed * torch.sin(theta) + speed * torch.cos(theta) * u_beta
#         theta_dot = speed/rear_dist * u_beta
#         speed_dot = u_a
#         dzdt = torch.stack([x_dot, y_dot, theta_dot, speed_dot, \
#                             torch.tensor(0.0, device=z.device), \
#                             torch.tensor(0.0, device=z.device)])    # 这里的dxdt 是一个 6 维的 tensor，前面四维是状态的导数，后面两维是控制输入的导数（因为控制输入在这个模型里是直接给定的，所以它们的导数是0）
#         dzdt = dzdt.unsqueeze(0)
#         return dzdt

def train_dynamics(z, u):
    theta, speed = z[:, 2], z[:, 3]
    u_a, u_s = u[:, 0], u[:, 1]
    x_dot = speed * torch.cos(theta) - speed * torch.sin(theta) * u_s
    y_dot = speed * torch.sin(theta) + speed * torch.cos(theta) * u_s
    theta_dot = speed / rear_dist * u_s
    speed_dot = u_a
    return torch.stack([x_dot, y_dot, theta_dot, speed_dot], dim=1)

def train_rk4(z, u):
    k1 = train_dynamics(z, u)
    k2 = train_dynamics(z + dt / 2 * k1, u)
    k3 = train_dynamics(z + dt / 2 * k2, u)
    k4 = train_dynamics(z + dt * k3, u)
    z_next = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z_next

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
    u_a, u_s = u[0], u[1]
    x_dot = speed * np.cos(theta) - speed * np.sin(theta) * u_s
    y_dot = speed * np.sin(theta) + speed * np.cos(theta) * u_s
    theta_dot = speed / rear_dist * u_s
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
    u_s = ca.SX.sym('u_s')

    # Define the Hamiltonian
    H = (r1*u_a**2 + r2*u_s**2 +
        lambda_x * (-speed * ca.sin(theta) * u_s) +
        lambda_y * ( speed * ca.cos(theta) * u_s) +
        lambda_theta * (speed/rear_dist * u_s) +
        lambda_speed * u_a
        )

    # Set up the QP problem
    qp = {
        'x': ca.vertcat(u_a, u_s),  # Decision variables [u_a, u_s]
        'f': H,                    # Cost function (Hamiltonian)
        'g': ca.vertcat()          # No additional equality/inequality constraints
    }

    # Set bounds for the decision variables

    lbx = [u_a_min, u_s_min]  # Lower bounds for [u_a, u_s]
    ubx = [u_a_max, u_s_max]  # Upper bounds for [u_a, u_s]

    opts = {
    'printLevel': 'none'  # Suppress solver output for qpoases
    }
    # # Create QP solver
    # S = ca.qpsol('S', 'qpoases', qp, opts)

    # # Solve the problem
    # solution = S(lbx=lbx, ubx=ubx)
    with open("./bin/qpoases_output.log", "a") as flog:
        with contextlib.redirect_stdout(flog), contextlib.redirect_stderr(flog):
            S = ca.qpsol('S', 'qpoases', qp, opts)
            solution = S(lbx=lbx, ubx=ubx)

    # Extract results
    u_a_opt = solution['x'][0]
    u_s_opt = solution['x'][1]
    return u_a_opt, u_s_opt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_animation(t_span, x_traj, y_traj, theta_traj, speed_traj, costate_trajectory, option):

    # Set up figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Left plot: State trajectory
    axs[0].set_xlabel("Time (s)",fontsize=20, fontweight='bold')
    axs[0].set_ylabel("State Trajectory",fontsize=20, fontweight='bold')
    line_state1, = axs[0].plot([], [], linestyle='-', dashes=[3, 1], label=r"$x_{ncr}$", linewidth=5)
    line_state2, = axs[0].plot([], [], linestyle='-', dashes=[3, 1], label=r"$y_{ncr}$", linewidth=5)
    line_state3, = axs[0].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\theta_{ncr}$", linewidth=5)
    line_state4, = axs[0].plot([], [], linestyle='-', dashes=[3, 1], label=r"$speed_{ncr}$", linewidth=5)  # 追加speed轨迹
    axs[0].legend(fontsize=28)
    axs[0].grid(True)
    axs[0].set_xlim(0, T_sim)
    axs[0].set_ylim(-10, 10)
    axs[0].tick_params(axis="both", labelsize=20)

    # Right plot: Co-state trajectory
    axs[1].set_xlabel("Time (s)", fontsize=20, fontweight='bold')
    axs[1].set_ylabel("Lambda values", fontsize=20, fontweight='bold')
    # Initialize co-state lines
    line_lambda1, = axs[1].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\lambda{x}$", linewidth=5)
    line_lambda2, = axs[1].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\lambda{y}$", linewidth=5)
    line_lambda3, = axs[1].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\lambda{\theta}$", linewidth=5)
    line_lambda4, = axs[1].plot([], [], linestyle='-', dashes=[3, 1], label=r"$\lambda{speed}$", linewidth=5)  # 追加speed的costate轨迹
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
        line_state4.set_data(t_span[:frame+1], speed_traj[:frame+1])  # 追加speed轨迹的更新

        # # Update co-state trajectory plot
        n = len( costate_trajectory[frame][0, :, 0])
        t_span_costate = np.linspace(frame*dt, (frame+n-1)*dt, n)
        axs[1].set_xlim(frame*dt, (frame+n-1)*dt)
        line_lambda1.set_data(t_span_costate, costate_trajectory[frame][0, :, 0])
        line_lambda2.set_data(t_span_costate, costate_trajectory[frame][0, :, 1])
        line_lambda3.set_data(t_span_costate, costate_trajectory[frame][0, :, 2])
        line_lambda4.set_data(t_span_costate, costate_trajectory[frame][0, :, 3])  # 追加speed的costate轨迹的更新

        return line_state1, line_state2, line_state3, line_state4, line_lambda1, line_lambda2, line_lambda3, line_lambda4

    # Create animation
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=total_steps_sim, interval=100, blit=True)
    output_dir = f"./bi_figs/bi_animation_{option}.gif"
    ani.save(output_dir, writer=animation.PillowWriter(fps=20))
    print(f"Animation saved to {output_dir}")




# ==============================================================================
# ==============================================================================
# ==================== New animation function for bicycle =====================
# ==============================================================================
# ==============================================================================

# note: we need u_beta to calculate the heading of the bicycle
def save_animation_bicycle_trajectory(x_robot, y_robot, theta_robot, speed_robot, u_s_robot, initial_state_option, gif_name, start_xy=None, goal_xy=None, obstacles=None,
                                       robot_r=0.25, margin=0.05):
    os.makedirs("./bi_animation/bicycle", exist_ok=True)
    # if initial_state_option == 'a':
       
    #     start_pos = [-1.9, 2.0, -1.79, 0.0]
    # else:
    #     start_pos = [-5.24, 4.11, 2.72, 0.0]

    # if initial_state_option == 'c':
    #     end_pos = [1, 1, 0, 0]
    # else:
    #     end_pos = [0, 0, 0, 0]

    if start_xy is None:
        start_xy = (float(x_robot[0]), float(y_robot[0]))

    if goal_xy is None:
        # goal_xy = (float(x_robot[-1]), float(y_robot[-1]))
        if initial_state_option == 'c':
            goal_xy = (1, 1)
        else:
            goal_xy = (0, 0)

    # Create a figure for the animation
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(min(x_robot) - 1, max(x_robot) + 1)
    ax.set_ylim(min(y_robot) - 1, max(y_robot) + 1)
    ax.set_aspect('equal')  # Ensure equal scale for x and y axes
    ax.set_xlabel("X (m)", fontsize=20, fontweight='bold')
    ax.set_ylabel("Y (m)", fontsize=20, fontweight='bold')
    ax.tick_params(axis="both", labelsize=20)
    

   # Draw obstacles and safety boundaries 不管
    if obstacles is not None:
        for j, obs in enumerate(obstacles):
            xo = float(obs["xo"])
            yo = float(obs["yo"])
            r_obs = float(obs["r"])
            r_eff = r_obs + robot_r + margin
        # Physical obstacle
            ax.add_patch(plt.Circle((xo, yo), r_obs,fill=True, linewidth=3, color='black', label="Obstacle" if j == 0 else None, zorder=4))

        # Safety boundary enforced by CBF/HOCBF
            ax.add_patch(plt.Circle((xo, yo), r_eff, fill=False, linewidth=2, linestyle='--', color='red', label="Safety boundary" if j == 0 else None, zorder=3))

    # assume lr = lf
    # front_dist = rear_dist

    # Draw the robot components
    # robot_body = plt.Circle((0, 0), 0.25, color='cyan', fill=True, linewidth=2)  # Robot circular base   #was 0.25
    robot_body, = ax.plot([-rear_dist, 0], [0, 0], color="black", linewidth=3)
    wheel_width = 0.2*bi_scaling/2
    wheel_height = 0.6*bi_scaling/2

    wheelrear = plt.Rectangle((-rear_dist-wheel_height / 2, -wheel_width / 2), wheel_height, wheel_width, color='gold')  # Left wheel
    wheelfront = plt.Rectangle((         -wheel_height / 2, -wheel_width / 2), wheel_height, wheel_width, color='gold')  # Right wheel
    front_wheel_dot = plt.Circle((        wheel_height / 2, 0), radius=0.03, color='black')

    # Add the robot components to the plot
    # ax.add_patch(robot_body)     # 这里改成了plot，所以不需要add_patch了
    ax.add_patch(wheelrear)
    ax.add_patch(wheelfront)
    ax.add_patch(front_wheel_dot)
    
    # if i want to include heading arrow 
    # front_len = 0.35
    # front_arrow, = ax.plot([], [], color='black', linewidth=3, solid_capstyle='round')


    # Add the black solid circles to represent obstacles
    # start_marker_xy = plt.Circle((start_pos[0], start_pos[1]), 0.2, color='green', fill=True, label="Start")
    # goal_marker_xy = plt.Circle((end_pos[0], end_pos[1]), 0.2, color='red', fill=True, label="Goal")
    start_marker_xy = plt.Circle(start_xy, 0.2*bi_scaling, color='green', fill=True, label="Start", zorder=-10)
    goal_marker_xy = plt.Circle(goal_xy, 0.2*bi_scaling, color='red', fill=True, label="Goal", zorder=-10)
    ax.add_patch(start_marker_xy)
    ax.add_patch(goal_marker_xy)
    # Add a dynamic trajectory line
    robot_path, = ax.plot([], [], linestyle="--", color="saddlebrown", label="Traveled Path", linewidth=2)
    ax.legend(loc="upper right",fontsize=20)


    delta_robot = np.atan(tot_dist/rear_dist * u_s_robot)
    if len(delta_robot) == len(x_robot) - 1:
        delta_robot = np.append(delta_robot, delta_robot[-1])

    front_dist = tot_dist - rear_dist

    # Animation function
    def update(frame):
        # Update the robot's position and orientation
        x, y, theta, delta = x_robot[frame], y_robot[frame], theta_robot[frame], delta_robot[frame]
        
        # Update robot body position
        robot_body.set_data([x - rear_dist * np.cos(theta), x + front_dist * np.cos(theta)],
                            [y - rear_dist * np.sin(theta), y + front_dist * np.sin(theta)])
        
        # Wheel offsets relative to the robot's center
        # wheel_offset = 0.3 # Distance of wheels from center  
        
        # update rear wheel position and orientation
        wheelrear_center_x = x - rear_dist * np.cos(theta)
        wheelrear_center_y = y - rear_dist * np.sin(theta)
        wheelrear.set_xy((
            wheelrear_center_x - wheel_height / 2 * np.cos(theta) + wheel_width / 2 * np.sin(theta),
            wheelrear_center_y - wheel_height / 2 * np.sin(theta) - wheel_width / 2 * np.cos(theta)
        ))
        wheelrear.angle = np.degrees(theta)

        # update front wheel position and orientation
        wheelfront_center_x = x + front_dist * np.cos(theta)
        wheelfront_center_y = y + front_dist * np.sin(theta)
        wheelfront.set_xy((
            wheelfront_center_x - wheel_height / 2 * np.cos(theta + delta) + wheel_width / 2 * np.sin(theta + delta),
            wheelfront_center_y - wheel_height / 2 * np.sin(theta + delta) - wheel_width / 2 * np.cos(theta + delta)
        ))
        wheelfront.angle = np.degrees(theta + delta)

        # update front wheel dot position
        front_wheel_dot.set_center((
            wheelfront_center_x + wheel_height / 2 * np.cos(theta + delta),
            wheelfront_center_y + wheel_height / 2 * np.sin(theta + delta)
        ))
        
        # Update the dynamic trajectory
        robot_path.set_data(x_robot[:frame], y_robot[:frame])  # Update only up to the current frame

        return robot_body, wheelrear, wheelfront, robot_path  
    # Create the animation
    time_per_step = 0.05 # in seconds, adjusted based on experiment results
    plt.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=len(x_robot), blit=True)
    output_dir = f"./bi_animation/bicycle/{gif_name}.mp4"
    anim.save(output_dir, writer="ffmpeg", fps=1/time_per_step)
    print(f"Animation (wheeled robot motion) saved to {output_dir}")

# ================== Final shot of bicycle trajectory with safety boundaries ==================
def save_bicycle_final_shot(x_robot, y_robot, u_beta_robot, file_name, start_xy=None, goal_xy=None, theta_robot=None, obstacles=None, robot_r=0.25, margin=0.05,
    show_safety_boundary=True, draw_robot=True, robot_alpha=0.45,):
    """
    Save a static snapshot of the executed trajectory with optional safety boundaries
    and a bicycle rendering at the final state.
    Optional start_xy/goal_xy force fixed markers and viewport inclusion.
    """
    os.makedirs("./bi_figs/bi_final_shot", exist_ok=True)

    x_robot = np.asarray(x_robot).reshape(-1)
    y_robot = np.asarray(y_robot).reshape(-1)
    if theta_robot is not None:
        theta_robot = np.asarray(theta_robot).reshape(-1)
        if len(theta_robot) != len(x_robot):
            raise ValueError("theta_robot must have the same length as x_robot/y_robot.")

    if len(x_robot) == 0 or len(y_robot) == 0:
        raise ValueError("x_robot and y_robot must be non-empty.")

    traj_start_xy = (float(x_robot[0]), float(y_robot[0]))
    end_xy = (float(x_robot[-1]), float(y_robot[-1]))
    end_theta = float(theta_robot[-1]) if theta_robot is not None else 0.0
    end_u_beta = float(u_beta_robot[-1]) if u_beta_robot is not None else 0.0
    end_delta = np.arctan(np.tan(end_u_beta)*(rear_dist + rear_dist)/rear_dist)
    start_marker_xy = (float(start_xy[0]), float(start_xy[1]), ) if start_xy is not None else traj_start_xy
    goal_marker_xy = (float(goal_xy[0]), float(goal_xy[1]), ) if goal_xy is not None else end_xy

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x_robot, y_robot, color="tab:blue", linewidth=2.5, label="Trajectory", zorder=3)

    x_candidates = [float(np.min(x_robot)), float(np.max(x_robot)),start_marker_xy[0], goal_marker_xy[0],]
    y_candidates = [float(np.min(y_robot)), float(np.max(y_robot)), start_marker_xy[1], goal_marker_xy[1],]

    if obstacles is not None:                 # don't care for now
        for j, obs in enumerate(obstacles):
            xo = float(obs["xo"])
            yo = float(obs["yo"])
            r_obs = float(obs["r"])
            r_eff = r_obs + robot_r + margin


            ax.scatter( [xo], [yo], color="gray",s=45, alpha=0.5,label="Obstacle" if j == 0 else "_nolegend_", zorder=1,)
            ax.add_patch(plt.Circle((xo, yo),r_obs,color="gray",alpha=0.8,label="_nolegend_",zorder=1,))
            if show_safety_boundary:
                ax.add_patch(plt.Circle( (xo, yo), r_eff, fill=False, linestyle="--", linewidth=1.5, color="red", alpha=0.9,
                    label="_nolegend_" if j == 0 else "_nolegend_", zorder=2,))

            visible_r = r_eff if show_safety_boundary else r_obs
            x_candidates.extend([xo - visible_r, xo + visible_r])
            y_candidates.extend([yo - visible_r, yo + visible_r])

    ax.scatter([start_marker_xy[0]], [start_marker_xy[1]], color="green", s=45, alpha=0.5, label="Start", zorder=4)
    ax.scatter([goal_marker_xy[0]], [goal_marker_xy[1]], color="red", s=45, alpha=0.5, label="Goal", zorder=4)
    ax.add_patch(plt.Circle((start_marker_xy[0], start_marker_xy[1]), 0.15, color="green", fill=True,alpha=0.5, label="_nolegend_", zorder=4))
    ax.add_patch(plt.Circle((goal_marker_xy[0], goal_marker_xy[1]), 0.15, color="red", fill=True,alpha=0.5, label="_nolegend_", zorder=4))

    front_dist = rear_dist

    if draw_robot:
        # Draw a simple bicycle footprint at the terminal state.
        # body = plt.Circle(end_xy, robot_r, color="cyan", ec="black", lw=1.2, alpha=robot_alpha, zorder=5)
        # ax.add_patch(body)
        robot_body, = ax.plot([end_xy[0] - rear_dist * np.cos(end_theta), end_xy[0] + front_dist * np.cos(end_theta)],
                            [end_xy[1] - rear_dist * np.sin(end_theta), end_xy[1] + front_dist * np.sin(end_theta)], color="black", linewidth=3)

        wheel_width = 0.2*bi_scaling
        wheel_height = 0.6*bi_scaling

        # rear wheel
        wheelrear_center_x = end_xy[0] - rear_dist * np.cos(end_theta)
        wheelrear_center_y = end_xy[1] - rear_dist * np.sin(end_theta)
        wheelrear = plt.Rectangle((
            wheelrear_center_x - wheel_height / 2 * np.cos(end_theta) + wheel_width / 2 * np.sin(end_theta),
            wheelrear_center_y - wheel_height / 2 * np.sin(end_theta) - wheel_width / 2 * np.cos(end_theta)
        ), wheel_height, wheel_width, color='gold', angle=np.degrees(end_theta))
        ax.add_patch(wheelrear)

        # front wheel
        wheelfront_center_x = end_xy[0] + front_dist * np.cos(end_theta)
        wheelfront_center_y = end_xy[1] + front_dist * np.sin(end_theta)
        wheelfront = plt.Rectangle((
            wheelfront_center_x - wheel_height / 2 * np.cos(end_theta + end_delta) + wheel_width / 2 * np.sin(end_theta + end_delta),
            wheelfront_center_y - wheel_height / 2 * np.sin(end_theta + end_delta) - wheel_width / 2 * np.cos(end_theta + end_delta)
        ), wheel_height, wheel_width, color='gold', angle=np.degrees(end_theta + end_delta))
        ax.add_patch(wheelfront)



        # wheel_length = 0.20
        # wheel_width = 0.08
        # wheel_offset = robot_r + 0.05

        # c = np.cos(end_theta)
        # s = np.sin(end_theta)
        # fwd = np.array([c, s], dtype=float)
        # lat = np.array([-s, c], dtype=float)

        # def _wheel_polygon(center_xy):
        #     center = np.array(center_xy, dtype=float)
        #     p1 = center - 0.5 * wheel_length * fwd - 0.5 * wheel_width * lat
        #     p2 = center + 0.5 * wheel_length * fwd - 0.5 * wheel_width * lat
        #     p3 = center + 0.5 * wheel_length * fwd + 0.5 * wheel_width * lat
        #     p4 = center - 0.5 * wheel_length * fwd + 0.5 * wheel_width * lat
        #     return np.vstack([p1, p2, p3, p4])

        # wheel_left_center = np.array(end_xy, dtype=float) + wheel_offset * lat
        # wheel_right_center = np.array(end_xy, dtype=float) - wheel_offset * lat

        # wheel_left = plt.Polygon(_wheel_polygon(wheel_left_center), closed=True,  color="gold", ec="black", lw=0.8,alpha=robot_alpha,zorder=6,)
        # wheel_right = plt.Polygon(_wheel_polygon(wheel_right_center),closed=True, color="gold", ec="black", lw=0.8, alpha=robot_alpha, zorder=6,)
        # ax.add_patch(wheel_left)
        # ax.add_patch(wheel_right)


    pad = 0.35
    ax.set_xlim(min(x_candidates) - pad, max(x_candidates) + pad)
    ax.set_ylim(min(y_candidates) - pad, max(y_candidates) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)", fontsize=20, fontweight='bold')
    ax.set_ylabel("Y (m)", fontsize=20, fontweight='bold')
    ax.tick_params(axis="both", labelsize=24)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="best", prop={"weight": "bold", "size": 20})
    plt.tight_layout()

    output_path = f"./bi_figs/bi_final_shot/{file_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Final-shot figure saved to {output_path}")
