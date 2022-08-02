import sys
import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append('ballbeam-gym')

from ballbeam_gym.envs.setpoint import BallBeamSetpointEnv


kwargs = {'timestep': 0.05, 
          'setpoint': 4,
          'beam_length': 5.0,
          'max_angle': 0.2,
          'init_velocity': 0.0,
          'action_mode': 'continuous'}

env = BallBeamSetpointEnv(**kwargs)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=10000)

obs = env.reset()

ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
episodic_reward = 0
for ep in range(1000):
    obs = obs.astype(np.float)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

    episodic_reward += rewards

   
    print("Episode * {} * Avg Reward is ==> {}, action = {}".format(ep, episodic_reward, action))
    avg_reward_list.append(episodic_reward)

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

