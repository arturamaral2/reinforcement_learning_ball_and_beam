import sys
import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append('ballbeam-gym')
from ballbeam_gym.envs.setpoint import BallBeamSetpointEnv

kwargs = {'timestep': 0.05, 
          'setpoint': 2,
          'beam_length': 5.0,
          'max_angle': 0.2,
          'init_velocity': 0.0,
          'action_mode': 'discrete'}

env = BallBeamSetpointEnv(**kwargs)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)

obs = env.reset()
while True:
    obs = obs.astype(np.float)
    action, _states = model.predict(obs,deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()