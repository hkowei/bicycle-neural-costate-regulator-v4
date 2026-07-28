import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from config import (dt, n, u_a_max, u_a_min, u_s_max, u_s_min, 
                    T_sim, total_steps_sim, rear_dist, tot_dist,
                     r1, r2, bi_scaling, gamma)
import casadi as ca
import torch
import random
import os
import contextlib

def train_dynamics(z, u):
    # dynamics for training 
    theta, speed = z[:, 2], z[:, 3]
    u_a, u_s = u[:, 0], u[:, 1]
    x_dot = speed * torch.cos(theta) - speed * torch.sin(theta) * u_s
    y_dot = speed * torch.sin(theta) + speed * torch.cos(theta) * u_s
    theta_dot = speed / rear_dist * u_s
    speed_dot = u_a
    return torch.stack([x_dot, y_dot, theta_dot, speed_dot], dim=1)

def train_rk4(z, u):
    # RK4 integration step for dynamics
    k1 = train_dynamics(z, u)
    k2 = train_dynamics(z + dt / 2 * k1, u)
    k3 = train_dynamics(z + dt / 2 * k2, u)
    k4 = train_dynamics(z + dt * k3, u)
    z_next = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z_next
    
def bicycle_dynamics(z, u):
    # dynamics for simulation
    theta, speed = z[0,2], z[0,3]
    u_a, u_s = u[0], u[1]
    x_dot = speed * np.cos(theta) - speed * np.sin(theta) * u_s
    y_dot = speed * np.sin(theta) + speed * np.cos(theta) * u_s
    theta_dot = speed / rear_dist * u_s
    speed_dot = u_a
    return np.array([x_dot, y_dot, theta_dot, speed_dot])

def rk4(z, u):
    k1 = bicycle_dynamics(z, u)
    k2 = bicycle_dynamics(z + dt / 2 * k1, u)
    k3 = bicycle_dynamics(z + dt / 2 * k2, u)
    k4 = bicycle_dynamics(z + dt * k3, u)
    z_next = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z_next

def bicycle_solve_qp(lambda_x, lambda_y, lambda_theta, lambda_speed, theta, speed):

    # Define decision variables
    u_a = ca.SX.sym('u_a')
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

    # Solve the problem; redirect the printing message of qp solver to a log file
    with open("./bin/qpoases_output.log", "a") as flog:
        with contextlib.redirect_stdout(flog), contextlib.redirect_stderr(flog):
            S = ca.qpsol('S', 'qpoases', qp, opts)
            solution = S(lbx=lbx, ubx=ubx)

    # Extract results
    u_a_opt = solution['x'][0]
    u_s_opt = solution['x'][1]
    return u_a_opt, u_s_opt

def velocity_cbf(u_a, speed):
    u_a_cbf = max(-gamma*(speed-0.00), u_a_min)
    u_a_cbf = min(u_a_cbf, u_a_max)
    u_a_cbf = max(u_a_cbf, u_a)
    return u_a_cbf

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
        line_state4.set_data(t_span[:frame+1], speed_traj[:frame+1])

        # # Update co-state trajectory plot
        n = len( costate_trajectory[frame][0, :, 0])
        t_span_costate = np.linspace(frame*dt, (frame+n-1)*dt, n)
        axs[1].set_xlim(frame*dt, (frame+n-1)*dt)
        line_lambda1.set_data(t_span_costate, costate_trajectory[frame][0, :, 0])
        line_lambda2.set_data(t_span_costate, costate_trajectory[frame][0, :, 1])
        line_lambda3.set_data(t_span_costate, costate_trajectory[frame][0, :, 2])
        line_lambda4.set_data(t_span_costate, costate_trajectory[frame][0, :, 3])

        return line_state1, line_state2, line_state3, line_state4, line_lambda1, line_lambda2, line_lambda3, line_lambda4

    # Create animation
    plt.tight_layout()
    frame_skip = int(0.05/dt)
    frame_indices = range(0, total_steps_sim, frame_skip)
    ani = animation.FuncAnimation(fig, update, frames=frame_indices, interval=100, blit=True)
    output_dir = f"./bi_figs/bi_animation_{option}.gif"
    ani.save(output_dir, writer=animation.PillowWriter(fps=20))
    print(f"Animation saved to {output_dir}")



def save_animation_bicycle_trajectory(x_robot, y_robot, theta_robot, speed_robot, u_s_robot, initial_state_option, gif_name, start_xy=None, goal_xy=None, obstacles=None,
                                       robot_r=0.25, margin=0.05):
    # Visualized animation for bicycle
    os.makedirs("./bi_animation/bicycle", exist_ok=True)

    if start_xy is None:
        start_xy = (float(x_robot[0]), float(y_robot[0]))

    if goal_xy is None:
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
    
    # Draw the robot components
    robot_body, = ax.plot([-rear_dist, 0], [0, 0], color="black", linewidth=3)
    wheel_width = 0.2*bi_scaling/2
    wheel_height = 0.6*bi_scaling/2

    wheelrear = plt.Rectangle((-rear_dist-wheel_height / 2, -wheel_width / 2), wheel_height, wheel_width, color='gold')  # Left wheel
    wheelfront = plt.Rectangle((         -wheel_height / 2, -wheel_width / 2), wheel_height, wheel_width, color='gold')  # Right wheel
    front_wheel_dot = plt.Circle((        wheel_height / 2, 0), radius=0.03, color='black')

    # Add the robot components to the plot
    ax.add_patch(wheelrear)
    ax.add_patch(wheelfront)
    ax.add_patch(front_wheel_dot)
    
    # Draw the start and goal markers
    start_marker_xy = plt.Circle(start_xy, 0.2*bi_scaling, color='green', fill=True, label="Start", zorder=-10)
    goal_marker_xy = plt.Circle(goal_xy, 0.2*bi_scaling, color='red', fill=True, label="Goal", zorder=-10)
    ax.add_patch(start_marker_xy)
    ax.add_patch(goal_marker_xy)

    # Add a dynamic trajectory line
    robot_path, = ax.plot([], [], linestyle="--", color="saddlebrown", label="Traveled Path", linewidth=2)
    ax.legend(loc="upper right",fontsize=20)

    delta_robot = np.arctan(tot_dist/rear_dist * u_s_robot)
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
    time_per_step = dt                                  # in seconds, adjusted based on experiment results
    # Skip frames to maintain the real simulation time step
    frame_skip = int(0.05/dt)
    frame_indices = range(0, len(x_robot), frame_skip)
    plt.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=frame_indices, blit=True)
    output_dir = f"./bi_animation/bicycle/{gif_name}.mp4"
    anim.save(output_dir, writer="ffmpeg", fps=1/(time_per_step*frame_skip))
    print(f"Animation (wheeled robot motion) saved to {output_dir}")
