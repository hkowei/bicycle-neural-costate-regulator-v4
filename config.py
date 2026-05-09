dt = 0.05 # Sampling time
n = 5 # Prediction hoirzon
N = n
epoch = 1
# Control input Constraints
u_a_min = -1
u_a_max = 1
u_beta_min = -0.1    # beta should be small
u_beta_max = 0.1
# Total time of simulation and time steps
T_sim = 10
total_steps_sim = int(T_sim/dt)
# Regularized co-state loss
beta = 0.1

rear_dist = 0.5 # Distance from rear axle to center of gravity, used in Bicycle dynamics

r1 = 1.0; r2 = 1.0; q1 = 10.0; q2 = 10.0; q3 = 10; q4 = 10; h = 50

CONN_HIDDEN_DIMS = [2, 2, 2]

