dt = 0.05 # Sampling time


# Control input Constraints
u_a_min = -2
u_a_max = 2
u_beta_min = -2    # beta should be small
u_beta_max = 2
# Total time of simulation and time steps
T_sim = 15*2
total_steps_sim = int(T_sim/dt)
# Regularized co-state loss
beta = 0.99

rear_dist = 0.5 # Distance from rear axle to center of gravity, used in Bicycle dynamics

r1 = 3.0; r2 = 1.0; q1 = 10.0; q2 = 10.0; q3 = 10; q4 = 10; h = 50*2

n = 30 # Prediction hoirzon
N = n
CONN_HIDDEN_DIMS = [64, 128, 64]
Nsample = 10 
Nsample1, Nsample2, Nsample3, Nsample4 = 10, 10, 10, 10
epoch = 50

batch_size = 100