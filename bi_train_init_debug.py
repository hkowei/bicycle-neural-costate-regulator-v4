import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchdiffeq import odeint # Use odeint for integration
import numpy as np
from bi_utils_debug import BicycleDynamics, set_seed
from config import dt, beta, rear_dist, CONN_HIDDEN_DIMS, q1, q2, q3, q4, r1, r2, n, h, epoch, Nsample
from torchdiffeq import odeint


# Step 2: Create Dataset Class
class InitialStateDataset(Dataset):
    def __init__(self, initial_states):
        # Store the initial states as a PyTorch tensor
        self.initial_states = torch.tensor(initial_states, dtype=torch.float32)

    def __len__(self):
        return len(self.initial_states)

    def __getitem__(self, idx):   # 这里的getitem的名字规定好的，不能随便改
        # Return a single state
        return self.initial_states[idx]

# Neural Network Model
class CoNN(nn.Module):
    def __init__(self, prediction_horizon):
        super(CoNN, self).__init__()
        self.prediction_horizon = prediction_horizon
        h1, h2, h3 = CONN_HIDDEN_DIMS
        self.fc1 = nn.Linear(4, h1)                           # bicyle有四个状态 (之前是3)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, 4*prediction_horizon)        # bicycle有四个状态 (之前是3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, self.prediction_horizon, 4)   # 拆成 1, 30, 4 （之前是 1，30，3）
        return x


# Training Setup
def train_network(initial_states, n, h, q1, q2, q3, q4, r1, r2, model_save_path, batch_size=1, epochs=50, lr=2e-4):

    dataset = InitialStateDataset(initial_states)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    # Time span for integration
    t_span = torch.tensor([0, dt], dtype=torch.float32, device=device)
    ode_solver = BicycleDynamics()

    # Initialize NN
    model = CoNN(n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    # Define cost matrices
    Q = torch.diag(torch.tensor([q1, q2, q3, q4], device=device))  # State cost 现在有四个变量
    R = torch.diag(torch.tensor([r1, r2], device=device))      # Control input cost
    H =  h*Q                                                   # Terminal cost，保留，不必修改

    for epoch in range(epochs):     
        epoch_loss = 0
        epoch_lambda_loss = 0
        # Iterate over all initial conditions
        for state_0 in dataloader:                               # state_0 是从dataloader取出来的临时变量，代表一个 batch 的初始状态，这里 batch_size 是 1，所以是 (x, y, theta, speed)。state_0 的 shape 是 (1, 4)，因为 dataloader 会自动把它变成一个 batch 的形式，即使 batch_size 是 1。

            optimizer.zero_grad()                                # 将模型的梯度清零，为当前 batch 的训练做准备
            state_0 = state_0.to(device)

            costate_traj_k_hat = model(state_0)   # Predicted co-state trajectory starting at time k
            state_k = state_0
            L_stage = 0
            L_terminal = 0
            lambda_cost = 0

            for i in range(n):                                                  # 迭代预测的每一个时间步，计算对应的控制输入和阶段成本
                lambdax_i = costate_traj_k_hat[0,i,0]
                lambday_i = costate_traj_k_hat[0,i,1]
                lambdatheta_i = costate_traj_k_hat[0,i,2]
                lambdaspeed_i = costate_traj_k_hat[0,i,3]
                theta_i = state_k[0, 2]                     # 取当前状态的角度 theta，因为state_k 是 1,4 的 tensor，所以 state_k[0,2] 就是 theta 的值，state_k[0,3]是speed的值
                speed_i = state_k[0, 3]

                # 查看公式3.1， 用costate来计算最优控制
                u_a_opt  = -0.5 * lambdaspeed_i/r1        
                u_beta_opt = -0.5/r2 * ( -lambdax_i * speed_i * torch.sin(theta_i)
                            + lambday_i * speed_i * torch.cos(theta_i) + lambdatheta_i * speed_i/rear_dist)
                u_opt = torch.cat([u_a_opt.unsqueeze(0), u_beta_opt.unsqueeze(0)], dim=0).unsqueeze(0)


                # Compute stage cost using matrices
                state_cost = state_k @ Q @ state_k.T   # Quadratic cost for state
                control_cost = u_opt @ R @ u_opt.T     # Quadratic cost for control inputs
                
                lambda_cost += torch.abs(lambdax_i) + torch.abs(lambday_i) + torch.abs(lambdatheta_i) + torch.abs(lambdaspeed_i)
                L_stage += state_cost + control_cost

                # Solve the initial value problem using odeint (Step simulation forward by dt)
                z_and_u = torch.cat([state_k, u_opt], dim=1).to(device)
                result = odeint(ode_solver, z_and_u, t_span, method='rk4')    # odesolver即是bicycle dynamics
                state_k = result[-1,:,:4]                               # 这里的state_k 是 1,4 的 tensor，代表下一时刻的状态。result只取前四位的状态变量

            # Compute L_terminal
            L_terminal = state_k @ H @ state_k.T
            # Backpropagation
            loss = L_stage + L_terminal + beta*lambda_cost
            loss.backward()                            # 如果之前没有写 optimizer.zero_grad()，那么每次调用 loss.backward() 的时候，梯度会累积起来，这样就会导致模型的参数更新不正确。
            optimizer.step()
            epoch_loss += loss.item()                  # item: tensor 变 scalar
            epoch_lambda_loss += lambda_cost.item()
            

        avg_loss = epoch_loss / len(dataset)
        avg_lambda_loss = epoch_lambda_loss / len(dataset)
        print(f"********Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.2f}, Lambda Loss {avg_lambda_loss:.2f}********")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':                 # 如果直接运行 train.py，就会执行下面的代码，进行训练；如果在其他文件 import train.py，则不会执行下面的代码
    seed = 0
    set_seed(seed)

    # n = 5 # Prediction horizon
    # h = 50 # Terminal cost coefficient
    # epoch = 1
    os.makedirs("./model", exist_ok=True)

    # Step 1: Generate 1000 combinations of (x, y, theta)
    # Nsample = 2
    x_range = np.linspace(-2, 2, Nsample)                  # -2 到 2，取Nsample个点
    y_range = np.linspace(-2, 2, Nsample)
    theta_range = np.linspace(-2, 2, Nsample)
    speed_range = np.linspace(-2, 2, Nsample)

    # Create a grid of all combinations
    x, y, theta, speed = np.meshgrid(x_range, y_range, theta_range, speed_range)
    initial_states = np.vstack([x.ravel(), y.ravel(), theta.ravel(), speed.ravel()]).T       # 重点：把三维的网格数据变成一个二维的数组，每一行是一个 (x, y, theta) 的组合，最终得到 1000 行，3 列的数组

    # Randomly shuffle the data set
    np.random.shuffle(initial_states)
    
    # Train the model
    # q1 = 10.0; q2 = 10.0; q3 = 10.0; q4 = 10.0; r1 = 1.0; r2 = 1.0    # may need to import from config later
    model_save_path = f"./model/bi_t0_ncr_N{n}_h{h}_seed_{seed}_e{epoch}.pth"
    train_network(initial_states, n, h, q1, q2, q3, q4, r1, r2, model_save_path, epochs=epoch, lr=1e-3)
