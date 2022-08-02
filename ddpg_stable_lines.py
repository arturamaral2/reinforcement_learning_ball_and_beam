import sys
import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append('ballbeam-gym')
from ballbeam_gym.envs.setpoint import BallBeamSetpointEnv

kwargs = {'timestep': 0.05, 
          'setpoint': 3,
          'beam_length': 5.0,
          'max_angle': 0.2,
          'init_velocity': 0.0,
          'action_mode': 'continuous'}

env = BallBeamSetpointEnv(**kwargs)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)

ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
episodic_reward = 0

obs = env.reset()

while True:
    obs = obs.astype(np.float)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

