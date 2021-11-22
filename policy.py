import os
from typing import overload
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym

import tensorflow as tf
import torch
import scipy
from tensorflow import keras

np.random.seed(0)
tf.random.set_seed(0)

from utils import *

class RandomPolicy:
    '''
    随机策略，向随机方向走动

    env：进行测试的环境
    '''
    def __init__(self, env) -> None:
        self.action_dim = env.action_dim

    def decide(self, observation):
        action = np.random.randint(self.action_dim)
        return action

class DQN:
    '''
    DQN算法
    '''
    def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.002,
                 replayer_capacity=10000, batch_size=32):
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
                done) 
        '''
        训练时进行学习
        '''
        # 存储经验
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
        self.evaluate_net.save_weights('./models/DQN/evaluate_net')
        self.target_net.save_weights('./models/DQN/target_net')
        self.replayer.save()

    def load(self):
        '''
        加载模型
        '''
        self.evaluate_net.load_weights('./models/DQN/evaluate_net')
        self.target_net.load_weights('./models/DQN/target_net')
        self.replayer.load()

class DoubleDQN(DQN):
    '''
    Double DQN算法
    '''
    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation,
                done) # 存储经验
        observations, actions, rewards, next_observations, dones = \
                self.replayer.sample(self.batch_size) # 经验回放
        next_eval_qs = self.evaluate_net.predict(next_observations)
        next_actions = next_eval_qs.argmax(axis=-1)
        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs[np.arange(next_qs.shape[0]), next_actions] 
        us = rewards + self.gamma * next_max_qs * (1. - dones)
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done:
            self.target_net.set_weights(self.evaluate_net.get_weights())

    def save(self):
        '''
        保存模型
        '''
        self.evaluate_net.save_weights('./models/DoubleDQN/evaluate_net')
        self.target_net.save_weights('./models/DoubleDQN/target_net')
        self.replayer.save('./models/DoubleDQN/replayer.csv')

    def load(self):
        '''
        加载模型
        '''
        self.evaluate_net.load_weights('./models/DoubleDQN/evaluate_net')
        self.target_net.load_weights('./models/DoubleDQN/target_net')
        self.replayer.load('./models/DoubleDQN/replayer.csv')

class QActorCritic:
    '''
    简单的执行者/评论者算法
    '''
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n = env.action_dim
        self.gamma = gamma
        self.discount = 1.
        
        self.actor_net = self.build_network(output_size=self.action_n,
                output_activation=tf.nn.softmax,
                loss=tf.losses.categorical_crossentropy,
                **actor_kwargs)
        self.critic_net = self.build_network(output_size=self.action_n,
                **critic_kwargs)
    
    def build_network(self, hidden_sizes, output_size, input_size=None,
                activation=tf.nn.relu, output_activation=None,
                loss=tf.losses.mse, learning_rate=0.01):
        model = keras.Sequential()
        for idx, hidden_size in enumerate(hidden_sizes):
            kwargs = {}
            if idx == 0 and input_size is not None:
                kwargs['input_shape'] = (input_size,)
            model.add(keras.layers.Dense(units=hidden_size,
                    activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size,
                activation=output_activation))
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model
    
    def decide(self, observation):
        probs = self.actor_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action
    
    def learn(self, observation, action, reward, next_observation,
            done, next_action=None):
        # 训练执行者网络
        x = observation[np.newaxis]
        u = self.critic_net.predict(x)
        q = u[0, action]
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)[0, action]
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor,
                    1e-6, 1.))
            loss_tensor = -self.discount * q * logpi_tensor
        grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)
        self.actor_net.optimizer.apply_gradients(zip(
                grad_tensors, self.actor_net.variables))
        
        # 训练评论者网络
        u[0, action] = reward
        if not done:
            q = self.critic_net.predict(
                    next_observation[np.newaxis])[0, next_action]
            u[0, action] += self.gamma * q
        self.critic_net.fit(x, u, verbose=0)
        
        if done:
            self.discount = 1.
        else:
            self.discount *= self.gamma
    
    def save(self):
        '''
        保存网络参数
        '''
        self.actor_net.save_weights('./models/AC/actor_net')
        self.critic_net.save_weights('./models/AC/critic_net')

    def load(self):
        '''
        加载网络模型参数
        '''
        self.actor_net.load_weights('./models/AC/actor_net')
        self.critic_net.load_weights('./models/AC/critic_net')

class AdvantageActorCritic(QActorCritic):
    '''
    优势执行者/评论者算法
    '''
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n = env.action_dim
        self.gamma = gamma
        self.discount = 1.

        self.actor_net = self.build_network(output_size=self.action_n,
                output_activation=tf.nn.softmax,
                loss=tf.losses.categorical_crossentropy,
                **actor_kwargs)
        self.critic_net = self.build_network(output_size=1,
                **critic_kwargs)
    
    def learn(self, observation, action, reward, next_observation, done):
        x = observation[np.newaxis]
        u = reward + (1. - done) * self.gamma * \
                self.critic_net.predict(next_observation[np.newaxis])
        td_error = u - self.critic_net.predict(x)
        
        # 训练执行者网络
        x_tensor = tf.convert_to_tensor(observation[np.newaxis],
                dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)[0, action]
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor,
                    1e-6, 1.))
            loss_tensor = -self.discount * td_error * logpi_tensor
        grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)
        self.actor_net.optimizer.apply_gradients(zip(
                grad_tensors, self.actor_net.variables)) # 更新执行者网络
        
        # 训练评论者网络
        self.critic_net.fit(x, u, verbose=0) # 更新评论者网络
        
        if done:
            self.discount = 1. # 为下一回合初始化累积折扣
        else:
            self.discount *= self.gamma # 进一步累积折扣
    
    def save(self):
        '''
        保存网络参数
        '''
        self.actor_net.save_weights('./models/AdvantageAC/actor_net')
        self.critic_net.save_weights('./models/AdvantageAC/critic_net')

    def load(self):
        '''
        加载网络模型参数
        '''
        self.actor_net.load_weights('./models/AdvantageAC/actor_net')
        self.critic_net.load_weights('./models/AdvantageAC/critic_net')

class ElibilityTraceActorCritic(QActorCritic):
    '''
    带资格迹的执行者/评论者方法
    '''
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99,
            actor_lambda=0.9, critic_lambda=0.9):
        observation_dim = env.observation_dim
        self.action_n = env.action_dim
        self.actor_lambda = actor_lambda
        self.critic_lambda = critic_lambda
        self.gamma = gamma
        self.discount = 1.

        self.actor_net = self.build_network(input_size=observation_dim,
                output_size=self.action_n, output_activation=tf.nn.softmax,
                **actor_kwargs)
        self.critic_net = self.build_network(input_size=observation_dim,
                output_size=1, **critic_kwargs)
        self.actor_traces = [np.zeros_like(weight) for weight in
                self.actor_net.get_weights()]
        self.critic_traces = [np.zeros_like(weight) for weight in
                self.critic_net.get_weights()]
    
    def learn(self, observation, action, reward, next_observation, done):
        q =  self.critic_net.predict(observation[np.newaxis])[0, 0]
        u = reward + (1. - done) * self.gamma * \
                self.critic_net.predict(next_observation[np.newaxis])[0, 0]
        td_error = u - q
        
        # 训练执行者网络
        x_tensor = tf.convert_to_tensor(observation[np.newaxis],
                dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.))
            logpi_pick_tensor = logpi_tensor[0, action]
        grad_tensors = tape.gradient(logpi_pick_tensor,
                self.actor_net.variables)
        self.actor_traces = [self.gamma * self.actor_lambda * trace +
                self.discount * grad.numpy() for trace, grad in
                zip(self.actor_traces, grad_tensors)]
        actor_grads = [tf.convert_to_tensor(-td_error * trace,
                dtype=tf.float32) for trace in self.actor_traces]
        actor_grads_and_vars = tuple(zip(actor_grads,
                self.actor_net.variables))
        self.actor_net.optimizer.apply_gradients(actor_grads_and_vars)
        
        # 训练评论者网络
        with tf.GradientTape() as tape:
            v_tensor = self.critic_net(x_tensor)
        grad_tensors = tape.gradient(v_tensor, self.critic_net.variables)
        self.critic_traces = [self.gamma * self.critic_lambda * trace +
                self.discount* grad.numpy() for trace, grad in
                zip(self.critic_traces, grad_tensors)]
        critic_grads = [tf.convert_to_tensor(-td_error * trace,
                dtype=tf.float32) for trace in self.critic_traces]
        critic_grads_and_vars = tuple(zip(critic_grads,
                self.critic_net.variables))
        self.critic_net.optimizer.apply_gradients(critic_grads_and_vars)
        
        if done:
            # 下一回合重置资格迹
            self.actor_traces = [np.zeros_like(weight) for weight
                    in self.actor_net.get_weights()]
            self.critic_traces = [np.zeros_like(weight) for weight
                    in self.critic_net.get_weights()]
            # 为下一回合重置累积折扣
            self.discount = 1.
        else:
            self.discount *= self.gamma

    def save(self):
        '''
        保存网络参数
        '''
        self.actor_net.save_weights('./models/ElibilityTraceAC/actor_net')
        self.critic_net.save_weights('./models/ElibilityTraceAC/critic_net')

    def load(self):
        '''
        加载网络模型参数
        '''
        self.actor_net.load_weights('./models/ElibilityTraceAC/actor_net')
        self.critic_net.load_weights('./models/ElibilityTraceAC/critic_net')
    
class SAC(QActorCritic):
    '''
    柔性执行者/评论者算法
    '''
    def __init__(self, env, actor_kwargs, critic_kwargs,
            gamma=0.99, alpha=0.2, net_learning_rate=0.1,
            replayer_capacity=1000, batches=1, batch_size=64):
        observation_dim = env.observation_dim
        self.action_n = env.action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.net_learning_rate = net_learning_rate
        
        self.batches = batches
        self.batch_size = batch_size
        self.replayer = ReplayerBuffer(replayer_capacity)
        
        def sac_loss(y_true, y_pred):
            """ y_true 是 Q(*, action_n), y_pred 是 pi(*, action_n) """
            qs = alpha * tf.math.xlogy(y_pred, y_pred) - y_pred * y_true
            return tf.reduce_sum(qs, axis=-1)
        
        self.actor_net = self.build_network(input_size=observation_dim,
                output_size=self.action_n, output_activation=tf.nn.softmax,
                loss=sac_loss, **actor_kwargs)
        self.q0_net = self.build_network(input_size=observation_dim,
                output_size=self.action_n, **critic_kwargs)
        self.q1_net = self.build_network(input_size=observation_dim,
                output_size=self.action_n, **critic_kwargs)
        self.v_evaluate_net = self.build_network(
                input_size=observation_dim, output_size=1, **critic_kwargs)
        self.v_target_net = self.build_network(
                input_size=observation_dim, output_size=1, **critic_kwargs)
        
        self.update_target_net(self.v_target_net, self.v_evaluate_net)
        
    def update_target_net(self, target_net, evaluate_net, learning_rate=1.):
        target_weights = target_net.get_weights()
        evaluate_weights = evaluate_net.get_weights()
        average_weights = [(1. - learning_rate) * t + learning_rate * e
                for t, e in zip(target_weights, evaluate_weights)]
        target_net.set_weights(average_weights)
        
    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation,
                done)
        
        if done:
            for batch in range(self.batches):
                # 经验回放
                observations, actions, rewards, next_observations, \
                        dones = self.replayer.sample(self.batch_size)
                
                pis = self.actor_net.predict(observations)
                q0s = self.q0_net.predict(observations)
                q1s = self.q1_net.predict(observations)
                
                # 训练执行者
                self.actor_net.fit(observations, q0s, verbose=0)
                
                # 训练评论者
                q01s = np.minimum(q0s, q1s)
                entropic_q01s = pis * q01s - self.alpha * \
                        scipy.special.xlogy(pis, pis)
                v_targets = entropic_q01s.sum(axis=-1)
                self.v_evaluate_net.fit(observations, v_targets, verbose=0)
                
                next_vs = self.v_target_net.predict(next_observations)
                q_targets = rewards + self.gamma * (1. - dones) * \
                         next_vs[:, 0]
                q0s[range(self.batch_size), actions] = q_targets
                q1s[range(self.batch_size), actions] = q_targets
                self.q0_net.fit(observations, q0s, verbose=0)
                self.q1_net.fit(observations, q1s, verbose=0)
                
                # 更新目标网络
                self.update_target_net(self.v_target_net,
                        self.v_evaluate_net, self.net_learning_rate)

    def save(self):
        '''
        保存网络参数
        '''
        self.actor_net.save_weights('./models/SAC/actor_net')
        self.q0_net.save_weights('./models/SAC/q0_net')
        self.q1_net.save_weights('./models/SAC/q1_net')
        self.v_evaluate_net.save_weights('./models/SAC/v_evaluate_net')
        self.v_target_net.save_weights('./models/SAC/v_target_net')

    def load(self):
        '''
        加载网络模型参数
        '''
        self.actor_net.load_weights('./models/SAC/actor_net')
        self.q0_net.load_weights('./models/SAC/q0_net')
        self.q1_net.load_weights('./models/SAC/q1_net')
        self.v_evaluate_net.load_weights('./models/SAC/v_evaluate_net')
        self.v_target_net.load_weights('./models/SAC/v_target_net')

def play_qlearning(env, policy, train=False, render=False):
    episode_reward = 0
    observations = env.reset()
    while True:
        if render:
            env.render(False)
        actions = []
        for i in range(env.agent_num):
            action = policy.decide(observations[i])
            actions.append(action)
        next_observations, rewards, dones = env.step(actions)
        episode_reward += np.sum(rewards)
        if train:
            for i in range(env.agent_num):
                policy.learn(observations[i], actions[i], rewards[i], next_observations[i],
                    dones[i])
        if True in dones:
            break
        observations = next_observations
    return episode_reward

def play_sarsa(env, policy, train=False, render=False):
    episode_reward = 0
    observations = env.reset()
    actions = []
    for i in range(env.agent_num):
        action = policy.decide(observations[i])
        actions.append(action)
    while True:
        if render:
            env.render()
        next_observations, rewards, dones, _ = env.step(actions)
        episode_reward += np.sum(rewards)
        if True in dones:
            if train:
                for i in range(env.agent_num):
                    policy.learn(observations[i], actions[i], rewards[i], next_observations[i],
                        dones[i])
            break
        next_actions = []
        for i in range(env.agent_num):
            next_action = policy.decide(next_observations[i])
            next_actions.append(next_action)
        if train:
            policy.learn(observations, actions, rewards, next_observations,
                    dones, next_actions)
        observations, actions = next_observations, next_actions
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