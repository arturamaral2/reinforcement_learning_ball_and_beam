import gym
import sys
sys.path.append('ballbeam-gym')
from ballbeam_gym.envs.setpoint import BallBeamSetpointEnv
import matplotlib.pyplot as plt
import numpy as np
# pass env arguments as kwargs
kwargs = {'timestep': 0.05, 
          'setpoint': 4,
          'beam_length': 5.0,
          'max_angle': 0.2,
          'init_velocity': 0.0,
          'action_mode': 'continuous'}

# create env
env = BallBeamSetpointEnv(**kwargs)

Kp = 2.0
Kd = 1.0

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
episodic_reward = 0
# simulate 1000 steps
for i in range(1000):   
    # control theta with a PID controller
    env.render()
    theta = Kp*(env.bb.x - env.setpoint) + Kd*(env.bb.v)
    obs, reward, done, info = env.step(theta)
    
    episodic_reward += reward
    ep_reward_list.append(episodic_reward)
    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(i, avg_reward))
    avg_reward_list.append(avg_reward)

    if done:
        env.reset()

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()