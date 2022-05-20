from NavigationEnv import NavigationEnv
from policy import DQN, DoubleDQN, RandomPolicy, play_qlearning
from utils import *

env = NavigationEnv(size = 15, block_num = 5, agent_num = 3, block_size = 2)
net_kwargs = {'hidden_sizes' : [64, 64], 'lr' : 0.01}
policy = DoubleDQN(env, net_kwargs, gamma=0.99)

policy.epsilon = 0
policy.load()

for i in range(10):
    print('--------------------------------')
    play_qlearning(env, policy)
    env.render(i)
print(env.arrive, env.collision, env.overtime)
# 163 20 17
# 285 13 2