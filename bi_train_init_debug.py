import os
from xml.parsers.expat import model
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchdiffeq import odeint # Use odeint for integration
import numpy as np
from bi_utils_debug import set_seed, train_rk4
from config import dt, betav42, beta_h, rear_dist, CONN_HIDDEN_DIMS, q1, q2, q3, q4, r1, r2, n, h1, h2, h3, h4, epoch, batch_size, Nsample1, Nsample2, Nsample3, Nsample4, x_bound, y_bound, theta_bound, speed_bound, lr, VERSION
from torchdiffeq import odeint
import time
import argparse

beta = betav42
if VERSION != 'v4.3':
    raise ValueError(f"Version mismatch: expected 'v4.2' but got {VERSION}")


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
def train_network(initial_states, n, h1, h2, h3, h4, q1, q2, q3, q4, r1, r2, model_save_path, checkpoint_path, batch_size=1, epochs=50, lr=2e-4, continue_training=False):

    dataset = InitialStateDataset(initial_states)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')  # Force CPU for debugging
    print("Using device:", device)
    # Time span for integration
    t_span = torch.tensor([0, dt], dtype=torch.float32, device=device)
    # ode_solver = BicycleDynamics()

    # Initialize NN
    model = CoNN(n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, threshold=5e-4, cooldown=2, min_lr=1e-5)
    old_epoch = 0
  
    if continue_training:
        with open("rundata.txt", "r") as f:
            old_rundata = f.read().strip()
        old_epoch = int(old_rundata.split("_e")[1].split("_")[0])
        print("old rundata:", old_rundata)
        print("old epoch:", old_epoch)
        print("total epoch:", epochs)
        print("epoch to train:", epochs - old_epoch)
        old_checkpoint_path = f"./checkpoint/bi_t0_ncr_N{n}_seed_{seed}_e{old_epoch}.pth"
        old_checkpoint = torch.load(old_checkpoint_path)
        model.load_state_dict(old_checkpoint["model_state_dict"])
        required_keys = ["epoch","model_state_dict","optimizer_state_dict","scheduler_state_dict",]
        for key in required_keys:
            if key not in old_checkpoint:
                raise KeyError(
                    f"Checkpoint is missing '{key}'. "
                    f"This file is probably from an old training version and cannot be resumed."
                )
        optimizer.load_state_dict(old_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(old_checkpoint["scheduler_state_dict"])
        old_epoch_from_checkpoint = old_checkpoint["epoch"]
        if old_epoch != old_epoch_from_checkpoint:
            error_msg = f"Epoch mismatch: checkpoint epoch {old_epoch_from_checkpoint} does not match epoch from rundata {old_epoch}"
            print(error_msg)
            raise ValueError(error_msg)
        # old_model_name = f'bi_t0_ncr_N{n}_seed_{seed}_e{old_epoch}.pth'
        # print(f'load model: {old_model_name}')
        # model.load_state_dict(torch.load(f'./model/{old_model_name}'))

    

    model.train()

    # Define cost matrices
    Q = torch.diag(torch.tensor([q1, q2, q3, q4], device=device))  # State cost 现在有四个变量
    R = torch.diag(torch.tensor([r1, r2], device=device))      # Control input cost
    # H =  h*Q                                                   # Terminal cost，保留，不必修改
    H = torch.diag(torch.tensor([h1, h2, h3, h4], device=device))

    epoch_start = time.time()

    epochs_to_train = epochs - old_epoch
    for epoch in range(epochs_to_train):   

        if epoch == 1:
            now = time.time()
            first_epoch_time = now - epoch_start
            est_tot = first_epoch_time * epochs
            print(
                f"First epoch runtime: {first_epoch_time/60:.2f} min, "
                f"estimated total time: {est_tot/60:.2f} min",
                flush=True
            )

        dyn_duration = 0
        back_prop_duration = 0
        
        epoch_loss = 0
        epoch_lambda_loss = 0
        # Iterate over all initial conditions
        # for state_0 in dataloader:                               # state_0 是从dataloader取出来的临时变量，代表一个 batch 的初始状态，这里 batch_size 是 1，所以是 (x, y, theta, speed)。state_0 的 shape 是 (1, 4)，因为 dataloader 会自动把它变成一个 batch 的形式，即使 batch_size 是 1。
        for batch_idx, state_0 in enumerate(dataloader):
            # if batch_idx % 200 == 0:
            #     now = time.time()
            #     elapsed = now - epoch_start
            #     interval = now - last_print_time
            #     print(
            #         f"epoch {epoch+1}/{epochs}, "
            #         f"batch {batch_idx}/{len(dataloader)}, "
            #         f"elapsed {elapsed:.1f}s, "
            #         f"dynamics {dyn_duration:.1f}s, "
            #         f"backprop {back_prop_duration:.1f}s, "
            #         f"last 200 batches {interval:.1f}s",
            #         flush=True
            #     )
            #     last_print_time = now
            #     dyn_duration = 0
            #     back_prop_duration = 0

            
            optimizer.zero_grad()                                # 将模型的梯度清零，为当前 batch 的训练做准备
            state_0 = state_0.to(device)

            costate_traj_k_hat = model(state_0)   # Predicted co-state trajectory starting at time k
            state_k = state_0
            L_stage = 0
            L_terminal = 0
            lambda_cost = 0
            lambdax_i = costate_traj_k_hat[:,0,0]
            lambday_i = costate_traj_k_hat[:,0,1]
            lambdatheta_i = costate_traj_k_hat[:,0,2]
            lambdaspeed_i = costate_traj_k_hat[:,0,3]

            for i in range(n):                                                  # 迭代预测的每一个时间步，计算对应的控制输入和阶段成本
                lambdax_i = costate_traj_k_hat[:,i,0]
                lambday_i = costate_traj_k_hat[:,i,1]
                lambdatheta_i = costate_traj_k_hat[:,i,2]
                lambdaspeed_i = costate_traj_k_hat[:,i,3]
                theta_i = state_k[:, 2]                     # 取当前状态的角度 theta，因为state_k 是 1,4 的 tensor，所以 state_k[0,2] 就是 theta 的值，state_k[0,3]是speed的值
                speed_i = state_k[:, 3]

                # 查看公式3.1， 用costate来计算最优控制
                u_a_opt = -0.5/r1 * lambdaspeed_i
                u_s_opt =  0.5/r2 * (lambdax_i * speed_i * torch.sin(theta_i) - lambday_i * speed_i * torch.cos(theta_i) - lambdatheta_i * speed_i / rear_dist) 
                # u_opt = torch.cat([u_a_opt_B.unsqueeze(0), u_beta_opt_B.unsqueeze(0)], dim=0).unsqueeze(0)
                u_opt = torch.stack([u_a_opt, u_s_opt], dim=1) # (B, 2))


                # Compute stage cost using matrices
                state_cost = (state_k @ Q * state_k).sum(dim=1)   # Quadratic cost for state
                control_cost = (u_opt @ R * u_opt).sum(dim=1)     # Quadratic cost for control inputs
                
                lambda_cost += torch.abs(lambdax_i) + torch.abs(lambday_i) + torch.abs(lambdatheta_i) + torch.abs(lambdaspeed_i)
                L_stage += state_cost + control_cost

                # Solve the initial value problem using odeint (Step simulation forward by dt)
                # dyn_start_time = time.time()
                # use_torchdiffeq = False
                # if use_torchdiffeq:
                #     z_and_u = torch.cat([state_k, u_opt], dim=1).to(device)
                #     result = odeint(ode_solver, z_and_u, t_span, method='rk4')    # odesolver即是bicycle dynamics
                #     # temporary comparison with handwritten RK4
                #     # if i == 0 and batch_idx % 50 == 0:
                #     #     state_new = train_rk4(state_k, u_opt)
                #     #     state_k = result[-1,:,:4]
                #     #     print("state_torchdiffeq =", state_k)
                #     #     print("state_myrk4 =", state_new)
                #     #     print("max diff =", torch.max(torch.abs(state_k - state_new)))
                #     state_k = result[-1,:,:4]                               # 这里的state_k 是 1,4 的 tensor，代表下一时刻的状态。result只取前四位的状态变量
                # else:
                state_k = train_rk4(state_k, u_opt)
                # dyn_end_time = time.time()
                # dyn_duration += dyn_end_time - dyn_start_time

            # Compute L_terminal
            L_terminal = (state_k @ H * state_k).sum(dim=1)
            L_terminal_costate = torch.abs(lambdax_i) + torch.abs(lambday_i) + torch.abs(lambdatheta_i) + torch.abs(lambdaspeed_i)
            # Backpropagation
            loss_B = L_stage + L_terminal + beta*lambda_cost + beta_h*L_terminal_costate
            loss = loss_B.mean()  # Average over the batch
            back_prop_start_time = time.time()
            loss.backward()                            # 如果之前没有写 optimizer.zero_grad()，那么每次调用 loss.backward() 的时候，梯度会累积起来，这样就会导致模型的参数更新不正确。
            back_prop_end_time = time.time()
            back_prop_duration += back_prop_end_time - back_prop_start_time
            optimizer.step()
            epoch_loss += loss_B.sum().item()                  # item: tensor 变 scalar
            epoch_lambda_loss += lambda_cost.sum().item()
            

        avg_loss = epoch_loss / len(dataset)
        avg_lambda_loss = epoch_lambda_loss / len(dataset)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"********Epoch [{epoch + 1 + old_epoch}/{epochs}], Loss: {avg_loss:.2f}, Lambda Loss {avg_lambda_loss:.2f}, Current Learning Rate: {current_lr:.2e}********")

    # Save the trained model
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Model saved to {model_save_path}")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':                 # 如果直接运行 train.py，就会执行下面的代码，进行训练；如果在其他文件 import train.py，则不会执行下面的代码
    seed = 0
    set_seed(seed)

    # n = 5 # Prediction horizon
    # h = 50 # Terminal cost coefficient
    # epoch = 1
    # os.makedirs("./model", exist_ok=True)
    os.makedirs("./checkpoint", exist_ok=True)

    # Step 1: Generate 1000 combinations of (x, y, theta)
    # Nsample = 2
    x_range = np.linspace(-x_bound, x_bound, Nsample1)                  # -2 到 2，取Nsample个点
    y_range = np.linspace(-y_bound, y_bound, Nsample2)
    theta_range = np.linspace(-theta_bound, theta_bound, Nsample3)
    speed_range = np.linspace(-speed_bound, speed_bound, Nsample4)

    # Create a grid of all combinations
    x, y, theta, speed = np.meshgrid(x_range, y_range, theta_range, speed_range)
    initial_states = np.vstack([x.ravel(), y.ravel(), theta.ravel(), speed.ravel()]).T       # 重点：把三维的网格数据变成一个二维的数组，每一行是一个 (x, y, theta) 的组合，最终得到 1000 行，3 列的数组

    # Randomly shuffle the data set
    np.random.shuffle(initial_states)
    
    # determine whether to continue training from a saved model
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue-training", action="store_true")
    args = parser.parse_args()
    continue_training = args.continue_training

    # Train the model
    # q1 = 10.0; q2 = 10.0; q3 = 10.0; q4 = 10.0; r1 = 1.0; r2 = 1.0    # may need to import from config later
    model_save_path = f"./model/bi_t0_ncr_N{n}_seed_{seed}_e{epoch}.pth"
    checkpoint_path = f"./checkpoint/bi_t0_ncr_N{n}_seed_{seed}_e{epoch}.pth"
    train_network(initial_states, n, h1, h2, h3, h4, q1, q2, q3, q4, r1, r2, model_save_path, checkpoint_path, batch_size=batch_size, epochs=epoch, lr=lr, continue_training=continue_training)
