import casadi as ca
import numpy as np
from config import (dt, T_sim, total_steps_sim, n, 
                    u_a_min, u_a_max, u_s_min, u_s_max,
                    rear_dist, h1, h2, h3, h4,
                    q1,q2,q3,q4,r1,r2,
                    x_ref,y_ref,theta_ref,speed_ref,
                    state_0a,state_0b)
from bi_utils_debug import bi_mpc_plot_traj, save_animation_bicycle_trajectory
import time

prediction_time = n * dt
# dt_mpc = 0.1
# dt_pred = 0.1
dtmpc_ratio = 5 #int(dt_mpc/dt)
dtpred_ratio = 5 # int(dt_pred/dt)
dt_mpc = dtmpc_ratio * dt
dt_pred = dtpred_ratio * dt
mpc_pred_horizon = int(prediction_time / dt_pred)
print(f'dt = {dt}\n'
      f'dt_mpc = {dt_mpc}\n'
      f'dt_pred = {dt_pred}\n'
      f'mpc prediction horizon = {mpc_pred_horizon}')

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
        # Define control decision variables (U only, for shooting method)
        U = ca.MX.sym("U", 2, self.mpc_pred_horizon)

        # Objective and cost weights
        obj = 0
        Q = np.diag([q1,q2,q3,q4]) * dtpred_ratio  # State cost weights
        R = np.diag([r1,r2]) * dtpred_ratio  # Control cost weights
        H = np.diag([h1,h2,h3,h4]) * dtpred_ratio

        # Constraints for control inputs only
        lbu = [self.u_a_min, self.u_s_min] * self.mpc_pred_horizon
        ubu = [self.u_a_max, self.u_s_max] * self.mpc_pred_horizon

        # Initial state
        z_k = current_state
        z_k = ca.reshape(z_k, -1, 1)
        # Build the objective by simulating forward using rk4 and accumulating cost
        for k in range(self.mpc_pred_horizon):
            # Get control input at step k
            u_k = U[:, k]
            ref_k = ref_traj[:, k]
            ref_k = ca.reshape(ref_k, -1, 1)

            # Compute the cost
            obj += ca.mtimes((z_k - ref_k).T, Q @ (z_k - ref_k)) + ca.mtimes(u_k.T, R @ u_k)

            # Forward propagate the state using rk4 with the control input
            z_k = self.rk4(z_k, u_k, dt_pred)
            
        # Add terminal cost
        obj += ca.mtimes((z_k - ref_traj[:, k+1]).T, H @ (z_k - ref_traj[:, k+1]))

        # Define the optimization problem
        nlp = {'f': obj, 'x': ca.reshape(U, -1, 1)}
        opts = {'ipopt.print_level':1,'print_time': 0,"ipopt.sb":"yes"}
        solver = ca.nlpsol('S', 'ipopt', nlp, opts)

        # Solve the optimization problem
        sol = solver(x0 = initial_input_guess, lbx=lbu, ubx=ubu)
        u_opt_traj = sol["x"].full()
        u = u_opt_traj[:2]
        return u_opt_traj  # Return only the first control action
    

def simulation(z_0, traj_ref):
    state_traj = [z_0]
    control_traj = []

    x_k = z_0
    # Simulation loop
    u_k = np.array([0,0])
    u_pred_traj = np.zeros((2 * mpc_pred_horizon,1))
    for k in range(total_steps_sim):
        # Calculate control input
        # ref_traj_segment = traj_ref[:, k:k + n + 1]
        ref_traj_segment = traj_ref[:, k : k + mpc_pred_horizon * dtpred_ratio + 1 : dtpred_ratio]
        if (k) % dtmpc_ratio == 0: 
            u_pred_traj = controller.get_control_input(x_k, ref_traj_segment, initial_input_guess=u_pred_traj)
            u_k = u_pred_traj[:2]
            print(f'MPC input update')
        print(f'MPC N={n} - timestep: {k} finished')
        
        control_traj.append(u_k)
        x_k = controller.rk4(x_k, u_k, dt)
        x_k = x_k.full().flatten().tolist()
        state_traj.append(x_k)
    # Convert state and control trajectories to numpy arrays for plotting
    state_traj = np.array(state_traj)
    control_traj = np.array(control_traj)
    return state_traj, control_traj

if __name__ == '__main__':

    # Change case to a, b or c here
    initial_state_option = 'c'
    if initial_state_option == 'a':
        state_0 = state_0a
    elif initial_state_option == 'b':
        state_0 = state_0b
    else:
        state_0 = state_0b
        # Define reference state
        x_ref = 1; y_ref = 1; theta_ref = 0
    
    
    t_span = np.arange(0, T_sim + n * dt, dt)
    traj_ref = np.zeros((4, total_steps_sim + n))
    if initial_state_option == 'c':
        traj_ref[0,:] = x_ref
        traj_ref[1,:] = y_ref
        traj_ref[2,:] = theta_ref
        traj_ref[3,:] = speed_ref

    # Initialize controller and initial state
    controller = bi_MPC(dt, dt_pred, n, mpc_pred_horizon, u_a_min, u_a_max, u_s_min, u_s_max)
    robot_x0 = state_0[0,0]; robot_y0 = state_0[0,1]; robot_theta0 = state_0[0,2]; robot_speed0 = state_0[0,3]
    x_0 = np.array([robot_x0, robot_y0, robot_theta0, robot_speed0])
    
    # Start timing
    start_time = time.time()
    state_traj, control_traj = simulation(x_0, traj_ref)
    # End timing
    end_time = time.time()
    
    # Compute total time taken
    execution_time = end_time - start_time
    time_per_step = execution_time / total_steps_sim
    print(f"Simulation executed in {execution_time:.2f}s, time per step: {time_per_step:.4f}s")

    # Plot state and control trajectories
    bi_mpc_plot_traj(state_mpc=state_traj, u_mpc=control_traj, 
              time=t_span, option=initial_state_option, dt_mpc=dt_mpc,dt_pred=dt_pred)
    
    x_mpc = state_traj[:, 0]
    y_mpc = state_traj[:, 1]
    theta_mpc = state_traj[:,2]
    speed_mpc = state_traj[:,3]
    u_a_mpc = control_traj[:, 0, 0]
    u_s_mpc = control_traj[:, 1, 0]

    # Compute trajectory gradients (numerical derivatives)
    x_traj_grad = np.gradient(x_mpc, dt)
    y_traj_grad = np.gradient(y_mpc, dt)
    theta_traj_grad = np.gradient(theta_mpc, dt)
    speed_traj_grad = np.gradient(speed_mpc, dt)

    u_a_traj_grad = np.gradient(u_a_mpc, dt)
    u_s_traj_grad = np.gradient(u_s_mpc, dt)

    # Compute mean squared derivative
    x_traj_msd = np.mean(x_traj_grad ** 2)
    y_traj_msd = np.mean(y_traj_grad ** 2)
    theta_traj_msd = np.mean(theta_traj_grad ** 2)
    speed_traj_msd = np.mean(speed_traj_grad ** 2)
    avg_state_msd = (x_traj_msd + y_traj_msd + theta_traj_msd + speed_traj_msd) / 4
    print(f"Average State Trajectory Mean Squared derivatives {avg_state_msd:.2f}")

    u_a_traj_msd = np.mean(u_a_traj_grad ** 2)
    u_s_traj_msd = np.mean(u_s_traj_grad ** 2)
    avg_u_msd = (u_a_traj_msd + u_s_traj_msd) / 2
    print(f"Average Control Input Trajectory Mean Squared derivatives {avg_u_msd:.2f}")

    # Calculate absolute convergence error
    if initial_state_option == 'c':
        abs_convergence_err = abs(x_mpc[-1] - x_ref) + abs(y_mpc[-1] - y_ref) + abs(theta_mpc[-1] - theta_ref) + abs(speed_mpc[-1] - speed_ref)
    else:
        abs_convergence_err = abs(x_mpc[-1]) + abs(y_mpc[-1]) + abs(theta_mpc[-1]) + abs(speed_mpc[-1])
    print(f'Final state: [{x_mpc[-1]:.2f}; {y_mpc[-1]:.2f}; {theta_mpc[-1]:.2f}; {speed_mpc[-1]:.2f}]')
    print(f'Absolute convergence error: {abs_convergence_err:.2f}')

    fig_name = f'bi_robot_animation_mpc_N{n}_dt{dt}_dtmpc{dt_mpc}_dtpred{dt_pred}_{initial_state_option}.gif'
    save_animation_bicycle_trajectory(x_robot=x_mpc, y_robot=y_mpc, theta_robot=theta_mpc, speed_robot=speed_mpc, u_s_robot=u_s_mpc, initial_state_option = initial_state_option, gif_name = fig_name, start_xy=None, goal_xy=None, obstacles=None,
                                     robot_r=0.25, margin=0.05)
 