import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym

import tensorflow as tf
import torch
from tensorflow import keras

np.random.seed(0)
tf.random.set_seed(0)

from utils import *

# DQN算法
class DQN:
    def __init__(self, env, net_kwargs={}, gamma=0.09, epsilon=0.001,
                 replayer_capacity=10000, batch_size=64):
        '''
        初始化
        net_kwargs:网络参数\\
        gamma：折扣因子\\
        epsilon：贪心算法\\
        replayer_capacity：经验回放容量\\
        batch_size：每次抽取的规模
        '''
        observation_dim = env.observation_dim
        self.action_dim = env.action_dim

        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.replayer = ReplayerBuffer(replayer_capacity)

        self.evaluate_net = self.build_network(input_size=observation_dim,
                        output_size=self.action_dim, **net_kwargs) # 评估网络
        self.target_net = self.build_network(input_size=observation_dim,
                        output_size=self.action_dim, **net_kwargs) # 目标网络

        self.target_net.set_weights(self.evaluate_net.get_weights())

    def build_network(self, input_size, hidden_sizes, output_size,
                        activation=tf.nn.relu, output_activation=None,
                        lr=0.01):
        '''
        构建评估网络和目标网络
        '''
        model = keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size,)) if not layer else {}
            model.add(keras.layers.Dense(units=hidden_size,
                    activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size,
                activation=output_activation)) # 输出层
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def decide(self, observation):
        '''
        决定要执行的动作
        '''
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            qs = self.evaluate_net.predict(observation[np.newaxis])
            action = np.argmax(qs)
        return action
        

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation,
                done) # 存储经验
        observations, actions, rewards, next_observations, dones = \
                self.replayer.sample(self.batch_size) # 经验回放
        
        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * (1. - dones) * next_max_qs
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done: # 更新目标网络
            self.target_net.set_weights(self.evaluate_net.get_weights())
    
    def save(self):
        '''
        保存模型
        '''
        self.evaluate_net.save_weights('./models/evaluate_net')
        self.target_net.save_weights('./models/target_net')
        self.replayer.save()

    def load(self):
        '''
        加载模型
        '''
        self.evaluate_net.load_weights('./models/evaluate_net')
        self.target_net.load_weights('./models/target_net')
        self.replayer.load()

def play_qlearning(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, next_observation,
                    done)
        if done:
            break
        observation = next_observation
    return episode_reward

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.observation_dim = env.observation_space.shape[0]
    env.action_dim = env.action_space.n
    net_kwargs = {'hidden_sizes' : [64, 64], 'lr' : 0.001}
    agent = DQN(env, net_kwargs=net_kwargs)

    # 训练
    episodes = 1
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play_qlearning(env, agent, train=True)
        episode_rewards.append(episode_reward)
        print(f'episode: {episode}, episode_reward:{episode_reward}')
        plot(episode_rewards)

    # 测试
    agent.save()
    agent.epsilon = 0. # 取消探索
    episode_rewards = [play_qlearning(env, agent) for _ in range(1)]
    print('平均回合奖励1 = {} / {} = {}'.format(sum(episode_rewards),
            len(episode_rewards), np.mean(episode_rewards)))

    test = DQN(env, net_kwargs=net_kwargs)
    test.epsilon = 0.
    test.load()
    episode_rewards = [play_qlearning(env, test) for _ in range(1)]
    print('平均回合奖励2 = {} / {} = {}'.format(sum(episode_rewards),
            len(episode_rewards), np.mean(episode_rewards)))
    print(test.target_net.get_weights()[1] == agent.target_net.get_weights()[1])