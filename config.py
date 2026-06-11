import numpy as np

VERSION = 'v4.4'

dt = 0.01 # Sampling time
dt_scale = dt / 0.05
# Total time of simulation and time steps
T_sim = 100
total_steps_sim = int(T_sim/dt)


# Training parameters
n = 500 # Prediction horizon
epoch = 100
betav42 = 0.99*dt_scale
beta_h = betav42*20
batch_size = 100

r1 = 30.0*dt_scale; r2 = 10*dt_scale; 
q1 = 10.0*dt_scale; q2 = 30.0*dt_scale; q3 = 4*dt_scale; q4 = 50*dt_scale; 
h1 = 600; h2 = 900; h3 = 700; h4 = 4000.0

Nsample1, Nsample2, Nsample3, Nsample4 = 10, 10, 10, 10
x_bound, y_bound, theta_bound, speed_bound = 2,2,2,2 #  sample range for initial states

CONN_HIDDEN_DIMS = [128, 512, 128]
lr = 1e-3   # starting learning rate
lr_factor = 0.5
lr_patience = 15
lr_threshold = 1e-4
lr_cooldown = 5
min_lr = 1e-5

bi_scaling = 1
rear_dist = 0.4*bi_scaling # important note: rear_dist right now is actually used as the length of the whole bicycle
tot_dist = rear_dist

# simulation
# Control input Constraints
u_a_min = -3
u_a_max = 3
u_s_min = -2    # beta should be small
u_s_max = 2
case = 'a'
state_0a = np.array([[-2, 2, -1.79, 0]])
state_0b = np.array([[-5.24, 4.11, 2.72, 0]])
x_ref = 1; y_ref = 1; theta_ref = 0; speed_ref = 0
