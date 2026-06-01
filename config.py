import numpy as np


dt = 0.05 # Sampling time
# Total time of simulation and time steps
T_sim = 10*2
total_steps_sim = int(T_sim/dt)


# Training parameters
n = 100 # Prediction hoirzon
N = n
epoch = 100
beta = 0.99
batch_size = 100

r1 = 1.0; r2 = 0.5; 
q1 = 8.0; q2 = 15.0; q3 = 5; q4 = 10; 
h1 = 2000.0*2; h2 = 1500.0*2; h3 = 2500.0*2; h4 = 300.0

Nsample1, Nsample2, Nsample3, Nsample4 = 10, 10, 10, 10
x_bound, y_bound, theta_bound, speed_bound = 2,2,2,2 #  sample range for initial states

CONN_HIDDEN_DIMS = [128, 512, 128]
lr = 1e-4

bi_scaling = 1
rear_dist = 0.4*bi_scaling # important note: rear_dist right now is actually used as the length of the whole bicycle


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
