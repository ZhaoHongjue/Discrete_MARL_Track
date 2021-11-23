import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    def save(self, path):
        '''
        保存学习得到的经验
        '''
        self.memory.to_csv(path)

    def load(self, path):
        self.memory = pd.read_csv(path, index_col=0)

class Chart:
    '''
    用于画图的类
    '''
    def __init__(self):
        self.figure, self.ax = plt.subplots(1, 1)
    
    def plot(self, episode_rewards):
        '''
        奖励曲线绘制函数
        '''
        self.ax.clear()
        self.ax.plot(episode_rewards)
        self.ax.set_xlabel('episode')
        self.ax.set_ylabel('reward')
        self.ax.set_title('episode rewards')
        self.figure.savefig('./curves/curve.png')



