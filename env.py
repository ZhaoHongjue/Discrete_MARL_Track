import numpy as np
from numpy.core.numeric import count_nonzero
from agent import Agent
from UI import Maze
np.random.seed(0)

class NavigationEnv:
    def __init__(self, size = 20, agent_num = 1, block_num = 10, block_size = 3, is_training = False):
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
        
        # 导航任务参数
        self.agent_num = agent_num
        self.block_num = block_num
        self.block_size = block_size
        self.is_training = is_training
        self.social_collision = False
        
        # 测试参数
        self.arrive = 0
        self.collision = 0
        self.overtime = 0
        self.social = 0

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
            self.map[self.agents[i].pos[0], self.agents[i].pos[1]] = 2
            self.map[self.agents[i].global_goal[0], self.agents[i].global_goal[1]] = 3

    def Agents_Observe(self):
        '''
        让环境中的每个机器人进行观测，观测以自身为中心的5*5的矩阵信息，之后展平，并把目标点在局部坐标系的位置加入其中
        '''
        observations = []
        map_fill = np.ones((self.size + 4, self.size + 4))
        map_fill[2:-2, 2:-2] = self.map

        for i in range(len(self.agents)):
            # 如果机器人已经结束工作，则直接令观测为-1
            if self.agents[i].over:
                observe = np.ones(self.observation_dim) * -1
            # 如果机器人没有结束
            else:
                observe = np.zeros((5, 5))
                pos = self.agents[i].pos + 2
                observe = map_fill[pos[0]-2:pos[0]+3, pos[1]-2:pos[1]+3].flatten().tolist()
                observe.pop(12) # 删除自身位置
                
                observe.append(self.agents[i].local_goal[0] / self.size)
                observe.append(self.agents[i].local_goal[1] / self.size)
                observe = np.asarray(observe, dtype = float)
            observations.append(observe)
        
        return observations
    
    def Check_Arrive(self):
        '''
        检测机器人是否到目标点
        '''
        for i in range(len(self.agents)):
            # 如果已经结束运行则跳过
            if self.agents[i].over:
                continue
            if self.agents[i].local_goal[0] == 0 and \
                self.agents[i].local_goal[1] == 0:
                self.agents[i].done_arrive = True
                print(f'{i} arrive!')
                if not self.is_training:
                    self.arrive += 1

    def Check_Collision(self):
        '''
        机器人的碰撞检测
        '''
        for i in range(len(self.agents)):
            # 若已经结束运行则跳过
            if self.agents[i].over:
                continue

            # 检测机器人是否有没有越过边界
            if not (0 <= self.agents[i].pos[0] < self.size and 0 <= self.agents[i].pos[1] < self.size):
                self.agents[i].done_collision = True

            # 如果机器人当前位置有静态障碍物
            elif self.map[self.agents[i].pos[0], self.agents[i].pos[1]] == 1: 
                self.agents[i].done_collision = True
            
            # 如果机器人当前位置有其他机器人
            elif self.map[self.agents[i].pos[0], self.agents[i].pos[1]] == 2:
                self.social_collision = True
                print('social collision!')
                for j in range(self.agent_num):
                    if self.agents[i].pos[0] == self.agents[j].pos[0] or \
                        self.agents[i].pos[1] == self.agents[j].pos[1]:
                        self.agents[i].done_collision = True
                        self.agents[i].done_social = True

            # 若发生碰撞：
            if self.agents[i].done_collision:
                # 机器人的位置和之前相同
                self.agents[i].pos = self.agents[i].last_pos.copy()
                self.agents[i].local_goal = self.agents[i].global_goal - self.agents[i].pos
                print(f'{i} collision!')
                # 若未在训练
                if not self.is_training:
                    self.collision += 1
    
    def Check_over(self):
        '''
        检测机器人是否结束运行
        '''
        for i in range(len(self.agents)):
            if self.agents[i].done_collision or self.agents[i].done_arrive or self.agents[i].done_overtime:
                self.agents[i].over = True

    def Compute_Rewards(self):
        '''
        计算整体奖励
        '''
        rewards = []
        for agent in self.agents:
            if agent.over:
                reward = 0
            else:
                reward = agent.compute_reward()
            rewards.append(reward)
        return rewards
                    
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
        self.Check_over()
        
        if len(actions) != self.agent_num:
            print('step传入参数错误！')
            return

        rewards = []
        done_arrives = []
        done_collisions = []
        done_overtimes = []

        for i in range(len(self.agents)):
            # 如果该机器人已经结束则直接令reward为0
            if self.agents[i].over:
                pass
            else:
                # 执行对应动作
                self.agents[i].set_action(actions[i])
                self.agents[i].steps += 1

                # 判断是否超时
                if not self.is_training:
                    if self.agents[i].steps > 120:
                        self.agents[i].done_overtime = True
                        self.overtime += 1
        
                
        self.Check_Arrive()
        self.Check_Collision()

        observations = self.Agents_Observe()
        rewards = self.Compute_Rewards()

        done_arrives = [agent.done_arrive for agent in self.agents]
        done_collisions = [agent.done_collision for agent in self.agents]
        done_overtimes = [agent.done_overtime for agent in self.agents]
        
        dones = np.asarray(done_arrives, dtype = bool) + np.asarray(done_collisions, dtype = bool) \
            + np.asarray(done_overtimes, dtype = bool)

        self.Agents_Place_Refresh()
        
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
        pass
           
    def close(self):
        '''
        关闭图形化界面
        '''
        self.maze.destroy()
        pass

if __name__ == '__main__':
    env = NavigationEnv(size = 5, block_num = 6, agent_num=2, block_size=2)
    
    
    env.render(done=False)
    while True:
        observations, rewards, dones = env.step([7, 1])
        print(f'observations: {observations}')
        print(f'rewards: {rewards}')
        done = True in dones
        env.render(done)
        print('-------------------------------------------------')
        if done:
            env.reset()