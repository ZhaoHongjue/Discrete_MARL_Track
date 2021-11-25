import numpy as np
from env import NavigationEnv
from policy import *
from utils import Chart

env = NavigationEnv(size = 10, block_num = 3, agent_num = 3, block_size = 2, is_training = True)
net_kwargs = {'hidden_sizes' : [64, 64], 'lr' : 0.008}
policy = DoubleDQN(env, net_kwargs, gamma=0.99)
# policy.load()
chart = Chart()

# actor_kwargs = {'hidden_sizes' : [64,], 'learning_rate' : 0.005}
# critic_kwargs = {'hidden_sizes' : [64,], 'learning_rate' : 0.005}
# policy = SAC(env, actor_kwargs=actor_kwargs,
#         critic_kwargs=critic_kwargs, batches=50)

episodes = 10000
episode_rewards = []
draw = []
for episode in range(episodes):
    episode_reward = play_qlearning(env, policy, train = True, render=False)
    print('episode: {}, episode_reward: {}, sum: {}'.format(episode, episode_reward, np.sum(episode_reward)))
    episode_rewards.append(np.sum(episode_reward))
    draw.append(np.clip(np.sum(episode_reward), -100, 100))
    chart.plot(draw)
    if episode % 10 == 0:
        policy.save()
        np.save('./models/DoubleDQN/learning.npy', episode_rewards)
    
policy.save()

policy.epsilon = 0.
env.is_training = False
episode_rewards = [play_qlearning(env, policy, render = True) for _ in range(100)]
np.save('./models/DoubleDQN/test.npy', episode_rewards)
print(env.arrive, env.collision, env.overtime)
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))