import numpy as np


dt = 0.05 # Sampling time
# Total time of simulation and time steps
T_sim = 10*2
total_steps_sim = int(T_sim/dt)


# Training parameters
n = 30 # Prediction hoirzon
N = n
epoch = 50
beta = 0.99
batch_size = 100

r1 = 1.0; r2 = 1.0; 
q1 = 15.0; q2 = 15.0; q3 = 10; q4 = 10; 
h1 = 1000.0; h2 = 1000.0; h3 = 1000.0; h4 = 500.0

Nsample1, Nsample2, Nsample3, Nsample4 = 10, 10, 10, 10
x_bound, y_bound, theta_bound, speed_bound = 2,2,2,2 #  sample range for initial states

CONN_HIDDEN_DIMS = [128, 512, 128]
lr = 1e-4

bi_scaling = 0.5
rear_dist = 0.4*bi_scaling # Distance from rear axle to center of gravity, used in Bicycle dynamics


# simulation
# Control input Constraints
u_a_min = -3
u_a_max = 3
u_omega_min = -2    # beta should be small
u_omega_max = 2
case = 'a'
state_0a = np.array([[-2, 2, -1.79, 0]])
state_0b = np.array([[-5.24, 4.11, 2.72, 0]])
x_ref = 1; y_ref = 1; theta_ref = 0; speed_ref = 0
