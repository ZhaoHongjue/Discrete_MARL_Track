import numpy as np
from env import NavigationEnv
from policy import DQN, RandomPolicy, play_qlearning
from utils import plot

env = NavigationEnv(size = 10, block_num = 3, agent_num=1,block_size=2)
net_kwargs = {'hidden_sizes' : [64, 64], 'lr' : 0.005}
policy = DQN(env, net_kwargs)
policy.load()

episodes = 500
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_qlearning(env, policy, train = True)
    print('episode: {}, episode_reward: {}'.format(episode, episode_reward))
    episode_rewards.append(episode_reward)
    plot(episode_rewards)

policy.save()