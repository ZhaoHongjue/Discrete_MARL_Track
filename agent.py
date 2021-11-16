import numpy as np

class Agent:
    def __init__(self, initpos = [0, 0], goal = [0, 0], ob_size = 5, map_size = 5, isenemy = False):
        '''
        初始化机器人

        initpos：机器人初始位置\\
        ob_size：最大观测距离\\
        isenemy：是否为我方机器人
        '''
        # 设置机器人的初始位置以及当前位置
        self.initpos = np.asarray(initpos, dtype = float)
        self.pos = np.asarray(initpos, dtype = float)

        # 设置机器人的目标点信息
        self.global_goal = np.zeros(2)
        self.local_goal = np.zeros(2)
        self.set_goal(goal)

        # 机器人观测
        self.observe = np.zeros((ob_size, ob_size))
        
        # 任务参数
        self.done_arrive = False
        self.done_collision = False
        self.map_size = map_size
        self.isenemy = isenemy
    
    def set_goal(self, goal):
        '''
        设置机器人需要去的位置

        goal:需要去的位置的全局目标点
        '''
        goal = np.asarray(goal, dtype = float)
        if goal.shape != (2,):
            print('set_goal传参错误!')
        else:
            self.global_goal = goal
            self.local_goal = self.global_goal - self.pos

    def set_action(self, action):
        '''
        选择机器人动作，用0~7表示可以去的方向，依次为：左上、上、右上、左、右、左下、下、右下
        '''
        action = int(action)
        if action not in list(range(8)):
            print('set_action传参错误！')
        else:
            choices = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0],
                                [1, 0], [-1, 1], [0, 1], [1, 1]])
            self.pos += choices[action]
            self.local_goal = self.global_goal - self.pos
    
    def get_state(self, map):
        '''
        获取机器人的状态，包括观测、目标点位置等，检测是否发生碰撞或者到达目标点
        '''
        # 获取机器人的观测信息
        # 检测机器人是否到达目标点
        if (self.pos == self.global_goal).all() == True:
            self.done_arrive = True
        # 检测机器人是否发生碰撞
        if self.observe[0, 0] == 4 or self.observe[0, 0] == 2 or \
                len(self.pos[self.pos < 0]) != 0 or len(self.pos[self.pos >= self.map_size]) != 0:
            self.done_collision = True
        return self.observe, self.done_arrive, self.done_collision

    def reset(self):
        '''
        重置机器人状态
        '''
        self.pos = self.initpos
        pass

    def compute_reward(self):
        '''
        计算当前机器人的奖励
        '''
        pass

if __name__ == '__main__':
    agent = Agent()
    goal = input('输入目标点：').split(' ')
    print('pos: {}'.format(agent.pos))
    while True:
        action = int(input('输入动作：'))
        agent.set_action(action)
        agent.set_goal(goal)
        print('pos: {}; global_goal: {}; local_goal: {}'.\
            format(agent.pos, agent.global_goal, agent.local_goal))
        if agent.get_state()[1]:
            print('到达目的地！')
            break
        if agent.get_state()[2]:
            print('碰撞！')
            break

         