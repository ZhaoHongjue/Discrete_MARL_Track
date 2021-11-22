from os import read
import numpy as np
from env import NavigationEnv
from policy import *
from utils import plot


def evaluate_policy(env, policy, render = False, episodes = 5):
    '''
    评估策略
    '''
    episode_rewards = []
    for _ in range(episodes):
        episode_reward = 0
        observations = env.reset()
        while True:
            actions = []
            for i in range(env.agent_num):
                action = policy.decide(observations[i])
                actions.append(action)
            observations, rewards, dones = env.step(actions)
            episode_reward += np.sum(rewards)
            done = True in dones
            if render:
                env.render(done)
            if done:
                episode_rewards.append(episode_reward)
                break
    print(episode_rewards)
    return np.mean(episode_rewards)
    

env = NavigationEnv(size = 10, block_num = 3, agent_num=1, block_size=2)
net_kwargs = {'hidden_sizes' : [64, 64], 'lr' : 0.01}
policy = DoubleDQN(env, net_kwargs, gamma=0.99)

# actor_kwargs = {'hidden_sizes' : [64,], 'learning_rate' : 0.005}
# critic_kwargs = {'hidden_sizes' : [64,], 'learning_rate' : 0.005}
# policy = SAC(env, actor_kwargs=actor_kwargs,
#         critic_kwargs=critic_kwargs, batches=50)

episodes = 1500
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_qlearning(env, policy, train = True, render=False)
    print('episode: {}, episode_reward: {}'.format(episode, episode_reward))
    if episode_reward >= -50:
        episode_rewards.append(episode_reward)
    plot(episode_rewards)
    if episode % 10 == 0:
        policy.save()
    
# episode_rewards = np.asarray(episode_rewards, dtype=float)
# episode_rewards = episode_rewards[episode_rewards >= -30]
# plot(episode_rewards)
policy.save()

# policy.epsilon = 0.
episode_rewards = [play_qlearning(env, policy, render=True) for _ in range(3)]
print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
        len(episode_rewards), np.mean(episode_rewards)))


# print('######################################')
# print(evaluate_policy(env, policy))
# print('######################################')

# random_policy = RandomPolicy(env)
# print(play_qlearning(env, random_policy))
# print('######################################')
# print(evaluate_policy(env, random_policy))
# print('######################################')