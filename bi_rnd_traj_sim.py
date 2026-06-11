from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from config import (x_bound, y_bound, theta_bound, speed_bound, 
                    total_steps_sim, CONN_HIDDEN_DIMS, n, epoch)
from bi_utils_debug import bicycle_solve_qp, rk4
from matplotlib.patches import Rectangle

x_lw_bound = 1
y_lw_bound = 1
theta_lw_bound = 0
x_up_bound = 7
y_up_bound = 7
theta_up_bound = 3.14
N_init_states = 200
hit = 0
x_ref = 0
y_ref = 0
theta_ref = 0
speed_ref = 0
converge_thre = 0.4

device = torch.device('cpu')
print(f'Using device: {device}')

# Neural Network Model
class CoNN(nn.Module):
    def __init__(self, prediction_horizon):
        super(CoNN, self).__init__()
        self.prediction_horizon = prediction_horizon
        h1, h2, h3 = CONN_HIDDEN_DIMS
        self.fc1 = nn.Linear(4, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, 4*prediction_horizon)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, self.prediction_horizon, 4)
        return x
seed = 0
np.random.seed(seed)
model = CoNN(n).to(device)
checkpoint_path = f"./checkpoint/bi_t0_ncr_N{n}_seed_{0}_e{epoch}.pth"
checkpoint = torch.load(checkpoint_path)
print(f'checkpoint loaded from: {checkpoint_path}')
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

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


fig, ax = plt.subplots(figsize=(8,6))
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

    state_0 = np.array([[x0, y0, theta0, speed0]])
    state_traj_undisturbed = [state_0]
    state_k = state_0
    # simulation
    for i in range(total_steps_sim):
        state_k_tensor = torch.tensor(state_k, dtype=torch.float32, device=device)
        error_k = np.array([[state_k[0,0] - x_ref, state_k[0,1] - y_ref, state_k[0,2] - theta_ref, state_k[0,3] - speed_ref]])
        error_k_tensor = torch.tensor(error_k, dtype=torch.float32, device=device)
        costate_traj_k_hat = model(error_k_tensor).cpu().detach().numpy()
        lambda_x,lambda_y,lambda_theta,lambda_speed=costate_traj_k_hat[0,0,:]
        u_a, u_s = bicycle_solve_qp(lambda_x=lambda_x,lambda_y=lambda_y,
                                    lambda_theta=lambda_theta,lambda_speed=lambda_speed,
                                    theta=state_k_tensor[0,2].cpu().detach().numpy(),
                                    speed=state_k_tensor[0,3].cpu().detach().numpy())
        u_s = float(u_s)
        u_a = float(u_a)
        u_k = np.array([u_a,u_s])
        state_k=rk4(state_k,u_k)
        state_traj_undisturbed.append(state_k)

    # plot the current trajectory
    state_traj_undisturbed = np.array(state_traj_undisturbed).squeeze(1)
    x_traj = state_traj_undisturbed[:,0]
    y_traj = state_traj_undisturbed[:,1]
    theta_traj = state_traj_undisturbed[:,2]
    speed_traj = state_traj_undisturbed[:,3]
    abs_convergence_err = abs(x_traj[-1] - x_ref) + abs(y_traj[-1] - y_ref) + abs(theta_traj[-1] - theta_ref) + abs(speed_traj[-1] - speed_ref)
    good_converge = False
    if abs_convergence_err <= converge_thre:
        good_converge = True
        hit = hit+1
    plot_bicycle_traj(ax=ax,x_robot=x_traj,y_robot=y_traj,x_ref=x_ref,y_ref=y_ref, good_converge=good_converge)
    print(f"state_0_{j} = np.array([[{x0:.2f}, {y0:.2f}, {theta0:.2f}, {speed0:.2f}]]) Convergence: {good_converge}")

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
output_dir = f"./bi_figs/rate{hit/N_init_states:.2f}_N{N_init_states}_seed{seed}_thre{converge_thre}_{x_lw_bound}-{x_up_bound}-{y_lw_bound}-{y_up_bound}.png"
plt.savefig(output_dir, dpi=300)
plt.close(fig)
print(f"Figure saved to {output_dir}")




