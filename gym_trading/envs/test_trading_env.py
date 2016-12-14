
import gym

#import gym_trading

import numpy as np

env = gym.make('trading-v0')

env.time_cost_bps = 0

Episodes=1

for _ in range(Episodes):
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample() # random
        observation, reward, done, info = env.step(action)
        print observation,reward,done,info
        if done:
            print reward
        
