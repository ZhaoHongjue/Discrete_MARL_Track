import numpy as np
from env import NavigationEnv
from policy import DQN, DoubleDQN, RandomPolicy, play_qlearning
from utils import plot

env = NavigationEnv(size = 10, block_num = 3, agent_num=1, block_size=2)
net_kwargs = {'hidden_sizes' : [64, 64], 'lr' : 0.005}
policy = DoubleDQN(env, net_kwargs)

policy.epsilon = 0
policy.load()

for _ in range(15):
    play_qlearning(env, policy, render=False)