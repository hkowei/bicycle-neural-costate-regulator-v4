import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchdiffeq import odeint # Use odeint for integration
import numpy as np
from utils import UnicycleDynamics, set_seed
from config import dt, beta
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
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)
        self.fc4 = nn.Linear(2, 3*prediction_horizon)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, self.prediction_horizon, 3)   # 拆成 1, 30, 3
        return x


# Training Setup
def train_network(initial_states, n, h, q1, q2, q3, r1, r2, model_save_path, batch_size=1, epochs=50, lr=2e-4):

    dataset = InitialStateDataset(initial_states)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    # Time span for integration
    t_span = torch.tensor([0, dt], dtype=torch.float32, device=device)
    ode_solver = UnicycleDynamics()

    # Initialize NN
    model = CoNN(n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    # Define cost matrices
    Q = torch.diag(torch.tensor([q1, q2, q3], device=device))  # State cost
    R = torch.diag(torch.tensor([r1, r2], device=device))      # Control input cost
    H =  h*Q                                                   # Terminal cost，保留，不必修改

    for epoch in range(epochs):     
        epoch_loss = 0
        epoch_lambda_loss = 0
        # Iterate over all initial conditions
        for state_0 in dataloader:                               # state_0 是从dataloader取出来的临时变量，代表一个 batch 的初始状态，这里 batch_size 是 1，所以是 (x, y, theta)。state_0 的 shape 是 (1, 3)，因为 dataloader 会自动把它变成一个 batch 的形式，即使 batch_size 是 1。

            optimizer.zero_grad()                                # 将模型的梯度清零，为当前 batch 的训练做准备
            state_0 = state_0.to(device)

            costate_traj_k_hat = model(state_0)   # Predicted co-state trajectory starting at time k
            state_k = state_0
            L_stage = 0
            L_terminal = 0
            lambda_cost = 0

            for i in range(n):                                                  # 迭代预测的每一个时间步，计算对应的控制输入和阶段成本
                lambda1_i = costate_traj_k_hat[0,i,0]
                lambda2_i = costate_traj_k_hat[0,i,1]
                lambda3_i = costate_traj_k_hat[0,i,2]
                theta_i = state_k[0, 2]                     # 取当前状态的角度 theta，因为state_k 是 1,3 的 tensor，所以 state_k[0,2] 就是 theta 的值

                v_opt = -0.5 * (lambda1_i * torch.cos(theta_i) + lambda2_i * torch.sin(theta_i))   # 根据PMP的最优控制律计算 v 和 w。这里写torch是为了让它们成为 tensor，这样后面计算损失的时候就可以自动求导了，如果直接写成数值的话，就无法求导了。
                w_opt = -0.5 * lambda3_i                                                           # 这里没有写torch，是因为 lambda3_i 本身就是一个 tensor，所以不需要再写 torch.tensor() 来转换了，直接用 lambda3_i 就可以了。
                u_opt = torch.cat([v_opt.unsqueeze(0), w_opt.unsqueeze(0)], dim=0).unsqueeze(0)    # 把 v 和 w 组合成一个 1,2 的 tensor，作为控制输入

                # Compute stage cost using matrices
                state_cost = state_k @ Q @ state_k.T   # Quadratic cost for state
                control_cost = u_opt @ R @ u_opt.T     # Quadratic cost for control inputs
                
                lambda_cost += torch.abs(lambda1_i) + torch.abs(lambda2_i) + torch.abs(lambda3_i)
                L_stage += state_cost + control_cost

                # Solve the initial value problem using odeint (Step simulation forward by dt)
                z_and_u = torch.cat([state_k, u_opt], dim=1).to(device)
                result = odeint(ode_solver, z_and_u, t_span, method='rk4')
                state_k = result[-1,:,:3]                               # 这里的state_k 是 1,3 的 tensor，代表下一时刻的状态

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

    n = 5 # Prediction horizon
    h = 50 # Terminal cost coefficient
    epoch = 1
    os.makedirs("./model", exist_ok=True)

    # Step 1: Generate 1000 combinations of (x, y, theta)
    x_range = np.linspace(-2, 2, 3)                  # -2 到 2，取十个点
    y_range = np.linspace(-2, 2, 3)
    theta_range = np.linspace(-2, 2, 3)

    # Create a grid of all combinations
    x, y, theta = np.meshgrid(x_range, y_range, theta_range)
    initial_states = np.vstack([x.ravel(), y.ravel(), theta.ravel()]).T       # 重点：把三维的网格数据变成一个二维的数组，每一行是一个 (x, y, theta) 的组合，最终得到 1000 行，3 列的数组

    # Randomly shuffle the data set
    np.random.shuffle(initial_states)
    
    # Train the model
    q1 = 10.0; q2 = 10.0; q3 = 10.0; r1 = 1.0; r2 = 1.0
    model_save_path = f"./model/bi_t0_ncr_N{n}_h{h}_seed_{seed}_e{epoch}.pth"
    train_network(initial_states, n, h, q1, q2, q3, r1, r2, model_save_path, epochs=epoch, lr=1e-3)
