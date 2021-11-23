import numpy as np
from env import NavigationEnv
from policy import DQN, DoubleDQN, RandomPolicy, play_qlearning
from utils import *

env = NavigationEnv(size = 10, block_num = 3, agent_num=1, block_size=2)
net_kwargs = {'hidden_sizes' : [64, 64], 'lr' : 0.01}
policy = DoubleDQN(env, net_kwargs, gamma=0.99)

policy.epsilon = 0
policy.load()

for i in range(5):
    print(i, end=' ')
    play_qlearning(env, policy, render=True)
print(env.arrive, env.collision, env.overtime)
# 163 20 17