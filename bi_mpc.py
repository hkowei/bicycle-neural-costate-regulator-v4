import casadi as ca
import numpy as np
from mpc import MPC
from config import dt, T_sim, total_steps_sim, N, v_max, v_min, w_max, w_min
from utils import plot_traj
import time

class MPC:
    def __init__(self, dt, N, v_min, v_max, w_min, w_max):
        # System parameters
        self.dt = dt
        self.N = N
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max

    def dynamics(self, x, u):
        # Unicycle model dynamics
        theta = x[2]
        V, omega = u[0], u[1]
        x_dot = V * ca.cos(theta)
        y_dot = V * ca.sin(theta)
        theta_dot = omega
        return ca.vertcat(x_dot, y_dot, theta_dot)

    def rk4(self, x, u):
        # RK4 integration step for dynamics
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + self.dt / 2 * k1, u)
        k3 = self.dynamics(x + self.dt / 2 * k2, u)
        k4 = self.dynamics(x + self.dt * k3, u)
        x_next = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def get_control_input(self, current_state, ref_traj, h):
        # Define control decision variables (U only, for shooting method)
        U = ca.MX.sym("U", 2, self.N)

        # Objective and cost weights
        obj = 0
        Q = np.diag([10, 10, 10])  # State cost weights
        R = np.diag([1, 1])   # Control cost weights
        H = h*Q

        # Constraints for control inputs only
        lbu = [self.v_min, self.w_min] * self.N
        ubu = [self.v_max, self.w_max] * self.N

        # Initial state
        x_k = current_state
        x_k = ca.reshape(x_k, -1, 1)
        # Build the objective by simulating forward using rk4 and accumulating cost
        for k in range(self.N):
            # Get control input at step k
            u_k = U[:, k]
            ref_k = ref_traj[:, k]
            ref_k = ca.reshape(ref_k, -1, 1)

            # Compute the cost
            obj += ca.mtimes((x_k - ref_k).T, Q @ (x_k - ref_k)) + ca.mtimes(u_k.T, R @ u_k)

            # Forward propagate the state using rk4 with the control input
            x_k = self.rk4(x_k, u_k)
            
        # Add terminal cost
        obj += ca.mtimes((x_k - ref_traj[:, k+1]).T, H @ (x_k - ref_traj[:, k+1]))

        # Define the optimization problem
        nlp = {'f': obj, 'x': ca.reshape(U, -1, 1)}
        opts = {'ipopt.print_level':1,'print_time': 0}
        solver = ca.nlpsol('S', 'ipopt', nlp, opts)

        # Solve the optimization problem
        sol = solver(lbx=lbu, ubx=ubu)
        u_opt_traj = sol["x"].full()
        u = u_opt_traj[:2]
        return u  # Return only the first control action
    


def simulation(x_0, traj_ref, h):
    state_traj = [x_0]
    control_traj = []

    x_k = x_0
    # Simulation loop
    for k in range(total_steps_sim):
        # Calculate control input
        ref_traj_segment = traj_ref[:, k:k + N + 1]
        u_k = controller.get_control_input(x_k, ref_traj_segment, h)
        print(f'MPC N={N} - timestep: {k} finished')
        
        control_traj.append(u_k)
        x_k = controller.rk4(x_k, u_k)
        x_k = x_k.full().flatten().tolist()
        state_traj.append(x_k)
    # Convert state and control trajectories to numpy arrays for plotting
    state_traj = np.array(state_traj)
    control_traj = np.array(control_traj)
    return state_traj, control_traj

if __name__ == '__main__':
    state_0a = np.array([[-1.16, 1.37, -1.79]])
    state_0b = np.array([[-5.24, 4.11, 2.72]])
    state_0c = state_0b

    # Change case to a, b or c here
    initial_state_option = 'c'
    if initial_state_option == 'a':
        state_0 = state_0a
    elif initial_state_option == 'b':
        state_0 = state_0b
    else:
        state_0 = state_0c
        # Define reference state
        x_ref = 1; y_ref = 1; theta_ref = 0
    
    
    t_span = np.arange(0, T_sim + N * dt, dt)
    traj_ref = np.zeros((3, total_steps_sim + N))
    if initial_state_option == 'c':
        traj_ref[0,:] = x_ref
        traj_ref[1,:] = y_ref
        traj_ref[2,:] = theta_ref

    # Initialize controller and initial state
    controller = MPC(dt, N, v_min, v_max, w_min, w_max)
    robot_x0 = state_0[0,0]; robot_y0 = state_0[0,1]; robot_theta0 = state_0[0,2]
    x_0 = np.array([robot_x0, robot_y0, robot_theta0])
    h = 50 # Terminal cost coefficient
    
    # Start timing
    start_time = time.time()
    state_traj, control_traj = simulation(x_0, traj_ref, h)
    # End timing
    end_time = time.time()
    
    # Compute total time taken
    execution_time = end_time - start_time
    time_per_step = execution_time / total_steps_sim
    print(f"Simulation executed in {execution_time:.2f}s, time per step: {time_per_step:.4f}s")

    # Plot state and control trajectories
    plot_traj(state_mpc=state_traj, u_mpc=control_traj, 
              time=t_span, h=h, option=initial_state_option)
    
    x_mpc = state_traj[:, 0]
    y_mpc = state_traj[:, 1]
    theta_mpc = state_traj[:,2]
    v_mpc = control_traj[:, 0, 0]
    w_mpc = control_traj[:, 1, 0]

    # Compute trajectory gradients (numerical derivatives)
    x_traj_grad = np.gradient(x_mpc, dt)
    y_traj_grad = np.gradient(y_mpc, dt)
    theta_traj_grad = np.gradient(theta_mpc, dt)

    v_traj_grad = np.gradient(v_mpc, dt)
    w_traj_grad = np.gradient(w_mpc, dt)

    # Compute mean squared derivative
    x_traj_msd = np.mean(x_traj_grad ** 2)
    y_traj_msd = np.mean(y_traj_grad ** 2)
    theta_traj_msd = np.mean(theta_traj_grad ** 2)
    avg_state_msd = (x_traj_msd + y_traj_msd + theta_traj_msd) / 3
    print(f"Average State Trajectory Mean Squared derivatives {avg_state_msd:.2f}")

    v_traj_msd = np.mean(v_traj_grad ** 2)
    w_traj_msd = np.mean(w_traj_grad ** 2)
    avg_u_msd = (v_traj_msd + w_traj_msd) / 2
    print(f"Average Control Input Trajectory Mean Squared derivatives {avg_u_msd:.2f}")

    # Calculate absolute convergence error
    if initial_state_option == 'c':
        abs_convergence_err = abs(x_mpc[-1] - x_ref) + abs(y_mpc[-1] - y_ref) + abs(theta_mpc[-1] - theta_ref)
    else:
        abs_convergence_err = abs(x_mpc[-1]) + abs(y_mpc[-1]) + abs(theta_mpc[-1])
    print(f'Final state: [{x_mpc[-1]:.2f}; {y_mpc[-1]:.2f}; {theta_mpc[-1]:.2f}]')
    print(f'Absolute convergence error: {abs_convergence_err:.2f}')

