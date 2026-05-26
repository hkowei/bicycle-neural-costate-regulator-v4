import numpy as np


dt = 0.05 # Sampling time
# Total time of simulation and time steps
T_sim = 10*2
total_steps_sim = int(T_sim/dt)


# Training parameters
n = 30 # Prediction hoirzon
N = n
epoch = 70
beta = 0.01
batch_size = 100

r1 = 0.5; r2 = 0.5; 
q1 = 10.0; q2 = 20.0; q3 = 15; q4 = 10; 
h1 = 10.0; h2 = 10.0; h3 = 10.0; h4 = 10.0

Nsample1, Nsample2, Nsample3, Nsample4 = 10, 10, 10, 10
x_bound, y_bound, theta_bound, speed_bound = 2,2,2,2 #  sample range for initial states

CONN_HIDDEN_DIMS = [64, 128, 64]
lr = 1e-3

bi_scaling = 1
rear_dist = 0.5*bi_scaling # Distance from rear axle to center of gravity, used in Bicycle dynamics


# simulation
# Control input Constraints
u_a_min = -2
u_a_max = 2
u_omega_min = -2    # beta should be small
u_omega_max = 2
case = 'c'
state_0a = np.array([[-1.16, 1.37, -1.79, 0.5]])
state_0b = np.array([[-5.24, 4.11, 2.72, 0.5]])
x_ref = 1; y_ref = 1; theta_ref = 0; speed_ref = 0
