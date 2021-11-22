import numpy as np
from agent import Agent
from UI import Maze
np.random.seed(0)

class NavigationEnv:
    def __init__(self, size = 20, agent_num = 1, block_num = 10, block_size = 3):
        '''
        生成地图以及初始参数设置

        map中的0：可行通道；1：障碍物；2：我方机器人；3：目标点；4：敌方机器人；\\
        agent_num：机器人数量\\
        block_num：障碍块数量
        '''
        # 地图生成
        self.size = size
        self.map = np.zeros((size, size))
        self.agents = []

        self.agent_num = agent_num
        self.block_num = block_num
        self.block_size = block_size

        self.AddBlocks(self.block_num)
        self.AddAgents(self.agent_num)
        self.Agents_Place_Refresh()

        # 环境参数
        self.action_dim = 8 # 每个agent可以向周围八个方向运动
        self.observation_dim = 5 * 5 + 2 - 1 # 观测周围5 * 5的信息，并把局部坐标点输入

    def AddBlocks(self, num):
        '''
        在地图中添加障碍物

        num：障碍物个数
        '''
        for _ in range(num):
            p = [0.2, 0.8]
            block = np.random.choice([0,1], size = self.block_size**2, replace = True, p = p).reshape(self.block_size, -1)
            pos = np.random.randint(2, self.size - 2, size = (2,))
            self.map[pos[0]:pos[0]+self.block_size, pos[1]:pos[1]+self.block_size] = block

    def AddAgents(self, num):
        '''
        在地图中添加机器人

        num：机器人个数
        '''
        for _ in range(num):
            # 生成机器人初始位置
            while True:
                initpose = np.random.randint(self.size, size = (2,))
                if self.map[initpose[0], initpose[1]] == 0:
                    break

            # 生成机器人目标点
            while True:
                goal = np.random.randint(self.size, size = (2,)) 
                if self.map[goal[0], goal[1]] == 0:
                    break
            agent = Agent(initpos=initpose, goal=goal)
            self.agents.append(agent)

    def Agents_Place_Refresh(self):
        '''
        在地图上刷新机器人的位置和目标点
        '''
        # 使地图回到没有放置机器人的状态
        self.map[self.map == 2] = 0
        self.map[self.map == 3] = 0
        for i in range(len(self.agents)):
            # 如果机器人当前位置没有障碍物
            if self.map[self.agents[i].pos[0], self.agents[i].pos[1]] == 0 or \
                self.map[self.agents[i].pos[0], self.agents[i].pos[1]] == 3:
                self.map[self.agents[i].pos[0], self.agents[i].pos[1]] = 2
            # 否则则认为发生碰撞
            else:
                self.agents[i].done_collision = True
                self.agents[i].pos = self.agents[i].last_pos.copy()
                self.map[self.agents[i].pos[0], self.agents[i].pos[1]] = 2
            self.map[self.agents[i].global_goal[0], self.agents[i].global_goal[1]] = 3

    def Agents_Observe(self):
        '''
        让环境中的每个机器人进行观测，观测以自身为中心的5*5的矩阵信息，之后展平，并把目标点在局部坐标系的位置加入其中
        '''
        observations = []
        map_fill = np.ones((self.size + 4, self.size + 4))
        map_fill[2:-2, 2:-2] = self.map

        for i in range(self.agent_num):
            observe = np.zeros((5, 5))
            pos = self.agents[i].pos + 2
            observe = map_fill[pos[0]-2:pos[0]+3, pos[1]-2:pos[1]+3].flatten().tolist()
            observe.pop(12) # 删除自身位置
            
            observe.append(self.agents[i].local_goal[0] / self.size)
            observe.append(self.agents[i].local_goal[1] / self.size)
            observe = np.asarray(observe, dtype = float)
            observations.append(observe)
        
        return observations
        
    def reset(self):
        '''
        环境重置
        '''
        self.map = np.zeros((self.size, self.size))
        self.agents = []

        self.AddBlocks(self.block_num)
        self.AddAgents(self.agent_num)
        self.Agents_Place_Refresh()

        observations = self.Agents_Observe() 
        return observations
       
    def step(self, actions):
        '''
        env中的必备函数
        '''
        if len(actions) != self.agent_num:
            print('step传入参数错误！')
            return

        rewards = []
        done_arrives = []
        done_collisions = []
        done_overtimes = []

        for i in range(len(self.agents)):
            # 执行对应动作
            self.agents[i].set_action(actions[i])
            self.agents[i].steps += 1

            # 如果机器人的位置没有超越地图边界，则更新机器人位置
            if not (0 <= self.agents[i].pos[0] < self.size and 0 <= self.agents[i].pos[1] < self.size):
                self.agents[i].done_collision = True
                self.agents[i].pos = self.agents[i].last_pos.copy()    

            self.Agents_Place_Refresh()

            # 判断机器人是否到目标位置
            # print(f'local_goal: {self.agents[i].local_goal}')
            if self.agents[i].local_goal[0] == 0 and \
                self.agents[i].local_goal[1] == 0:
                self.agents[i].done_arrive = True

            # # 判断是否超时
            # if self.agents[i].steps > 120:
            #     self.agents[i].done_overtime = True

            # 计算回报
            reward = self.agents[i].compute_reward()

            done_arrives.append(self.agents[i].done_arrive)
            done_collisions.append(self.agents[i].done_collision)
            done_overtimes.append(self.agents[i].done_overtime)
            rewards.append(reward)

        observations = self.Agents_Observe()
        dones = np.asarray(done_arrives, dtype = bool) + np.asarray(done_collisions, dtype = bool) \
            + np.asarray(done_overtimes, dtype = bool)
        return observations, rewards, dones

    def render(self, done):
        '''
        绘制图形化界面

        done：True时会持续运行，False时会每个0.5秒重画一次
        '''
        # print(self.map)
        self.maze = Maze(self.map)
        if not done:
            self.maze.after(500, self.close)
        self.maze.mainloop()
           
    def close(self):
        '''
        关闭图形化界面
        '''
        self.maze.destroy()

if __name__ == '__main__':
    env = NavigationEnv(size = 10, block_num = 6, agent_num=1, block_size=2)
    
    
    env.render(done=False)
    while True:
        observations, rewards, dones = env.step([7])
        print(f'observations: {observations}')
        print(f'rewards: {rewards}')
        done = True in dones
        env.render(done)
        print('-------------------------------------------------')
        if done:
            env.reset()