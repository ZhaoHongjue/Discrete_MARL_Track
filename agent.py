import numpy as np

class Agent:
    def __init__(self, initpos = [0, 0], goal = [0, 0], isenemy = False):
        '''
        初始化机器人

        initpos：机器人初始位置\\
        ob_size：最大观测距离\\
        isenemy：是否为我方机器人
        '''
        # 设置机器人的初始位置以及当前位置
        self.initpos = np.asarray(initpos, dtype = int)  # 机器人初始位置，用于之后重置机器人
        self.pos = np.asarray(initpos, dtype = int)
        self.last_pos = np.asarray(initpos, dtype = int) # 机器人上次的位置，用于计算奖励

        # 设置机器人的目标点信息
        self.global_goal = np.zeros(2, dtype = int)
        self.local_goal = np.zeros(2, dtype = int)
        self.set_goal(goal)
        
        # 任务参数
        self.done_arrive = False
        self.done_collision = False
        self.done_overtime = False
        self.steps = 0
        self.isenemy = isenemy
    
    def set_goal(self, goal):
        '''
        设置机器人需要去的位置

        goal:需要去的位置的全局目标点
        '''
        goal = np.asarray(goal, dtype = int)
        if goal.shape != (2,):
            print('set_goal传参错误!')
        else:
            self.global_goal = goal
            self.local_goal = self.global_goal - self.pos

    def set_action(self, action):
        '''
        选择机器人动作，用0~7表示可以去的方向，依次为：左上、上、右上、左、右、左下、下、右下

        action：动作编号
        '''
        action = int(action)
        if action not in list(range(8)):
            print('set_action传参错误！')
        else:
            choices = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0],
                                [1, 0], [-1, 1], [0, 1], [1, 1]])
            self.last_pos = self.pos.copy()
            self.pos += choices[action]
            self.local_goal = self.global_goal - self.pos

    def reset(self):
        '''
        重置机器人状态
        '''
        self.pos = self.initpos.copy()
        self.last_pos = self.pos.copy()
        self.local_goal = self.global_goal - self.pos

        self.done_arrive = False
        self.done_collision = False
        self.done_overtime = False
        self.steps = 0

    def compute_reward(self):
        '''
        计算当前机器人的奖励
        '''
        if self.done_arrive:        # 成功到达
            reward = 20
            # print('arrive!')
        elif self.done_collision:   # 发生碰撞
            reward = -10
            # print('collision!')
        elif self.done_overtime:    # 运行超时
            reward = 0
            # print('overtime!')
        else:                       # 未结束
            reward1 = reward2 = reward3 = 0
            # 与目标点的距离缩短
            distance1 = np.sqrt((self.pos[0] - self.global_goal[0])**2 + (self.pos[1] - self.global_goal[1])**2)
            distance2 = np.sqrt((self.last_pos[0] - self.global_goal[0])**2 + (self.last_pos[1] - self.global_goal[1])**2) 
            # print('distance1: {:.3}, distance2: {:.3}'.format(distance1, distance2))
            reward1 = -0.5 * distance1 #0.1 * (distance2 - distance1)
            reward2 = -1 # 每走一步都消耗能量
            if self.steps >= 200:
                reward3 = -1
            else:
                reward3 = 0
            reward = reward1 + reward2 + reward3
        return reward

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
        if agent.local_goal.all() == 0:
            print('到达目的地！')
            agent.reset()

         