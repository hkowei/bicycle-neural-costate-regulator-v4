import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import (
    dt, T_sim, total_steps_sim,
    q1, q2, q3, q4,
    r1, r2,
    h1, h2, h3, h4,
    x_bound, y_bound, theta_bound, speed_bound,
    u_a_min, u_a_max, u_s_min, u_s_max,
    state_0a, state_0b,
    x_ref, y_ref, theta_ref, speed_ref,
)
from bi_utils_debug import rk4


class BicycleSACEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.dt = dt
        self.max_steps = total_steps_sim
        self.step_count = 0

        # observation/states space [x,y,theta,speed]
        self.observation_space = spaces.Box(
            low  = np.array([-x_bound,-y_bound,-theta_bound,-speed_bound],dtype=np.float32),
            high = np.array([ x_bound, y_bound, theta_bound, speed_bound],dtype=np.float32),
            dtype=np.float32,
        )

        # actions space [u_a,u_s]
        self.action_space = spaces.Box(
            low  = np.array([u_a_min,u_s_min],dtype=np.float32),
            high = np.array([u_a_max,u_s_max],dtype=np.float32),
            dtype=np.float32
        )

        self.state = None

    def reset(self, seed = None, options=None):
        super().reset(seed=seed)

        x0 = self.u