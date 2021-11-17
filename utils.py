import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym

import tensorflow as tf
import torch
from tensorflow import keras

np.random.seed(0)
tf.random.set_seed(0)

# 经验回放
class ReplayerBuffer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
            columns=['observation','action','reward','next_observation','done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        '''
        存储经验
        '''
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
    
    def sample(self, size):
        '''
        从经验池采样
        '''
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)
    def save(self):
        '''
        保存学习得到的经验
        '''
        self.memory.to_csv('./models/replayer.csv')

    def load(self):
        self.memory = pd.read_csv('./models/replayer.csv')
        self.count = self.memory.shape[0]
    
def plot(episode_rewards):
    '''
    奖励曲线绘制函数
    '''
    figure, ax = plt.subplots()
    ax.plot(episode_rewards)
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    ax.set_title('episode rewards')
    figure.savefig('./curves/curve.png')



