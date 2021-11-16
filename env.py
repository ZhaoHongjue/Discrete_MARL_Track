import numpy as np
from agent import Agent

class NavigationEnv:
    def __init__(self, size = 20, agent_num = 1, block_num = 8):
        '''
        生成地图
        map中的0：可行通道；1：障碍物；2：我方机器人；3：目标点；4：敌方机器人；
        agent_num：机器人数量
        block_num：障碍块数量
        '''
        self.size = size
        self.map = np.zeros((size, size))
        self.agents = []

        self.agent_num = agent_num
        self.block_num = block_num

        self.AddBlocks(self.block_num)
        self.AddAgents(self.agent_num)
        self.Agents_Place_Refresh()

    def AddBlocks(self, num):
        '''
        在地图中添加障碍物

        num：障碍物个数
        '''
        for _ in range(num):
            p = [0.2, 0.8]
            block = np.random.choice([0,1], size = 9, replace = True, p = p).reshape(3,3)
            pos = np.random.randint(2, self.size - 2, size = (2,))
            self.map[pos[0]:pos[0]+3, pos[1]:pos[1]+3] = block

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
            if self.map[self.agents[i].pos[0], self.agents[i].pos[1]] == 0:
                self.map[self.agents[i].pos[0], self.agents[i].pos[1]] = 2
            # 否则则认为发生碰撞
            else:
                self.agents[i].done_collision = True
            self.map[self.agents[i].global_goal[0], self.agents[i].global_goal[1]] = 3

    def Agents_Observe(self):
        '''
        让环境中的每个机器人进行观测，观测以自身为中心的5*5的矩阵信息，之后展平，并把目标点在局部坐标系的位置加入其中
        '''
        observations = []
        map_fill = np.ones((self.size + 4, self.size + 4))
        map_fill[2:-2, 2:-2] = self.map

        for agent in self.agents:
            observe = np.zeros((5, 5))
            pos = agent.pos + 2
            observe = map_fill[pos[0]-2:pos[0]+3, pos[1]-2:pos[1]+3].flatten().tolist()
            observe.append(agent.local_goal[0])
            observe.append(agent.local_goal[1])
            observe = np.asarray(observe, dtype = int)
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

        observations = self.Agents_Observe()
        rewards = []
        done_arrives = []
        done_collisions = []

        for i in range(len(self.agents)):
            # 执行对应动作
            self.agents[i].set_action(actions[i])

            # 如果机器人的位置没有超越地图边界，则更新机器人位置
            if 0 <= self.agents[i].pos[0] < 20 and 0 <= self.agents[i].pos[1] < 20:
                self.Agents_Place_Refresh()
            else:
                self.agents[i].done_collision = True

            # 判断机器人是否到目标位置
            # print(f'local_goal: {self.agents[i].local_goal}')
            if self.agents[i].local_goal[0] == 0 and \
                self.agents[i].local_goal[1] == 0:
                self.agents[i].done_arrive = True

            # 计算回报
            reward = self.agents[i].compute_reward()

            done_arrives.append(self.agents[i].done_arrive)
            done_collisions.append(self.agents[i].done_collision)
            rewards.append(reward)
        
        return observations, rewards, done_arrives, done_collisions

    def render(self):
        print(self.map)

if __name__ == '__main__':
    env = NavigationEnv(agent_num=1)
    
    while True:
        env.render()
        observations, rewards, done_arrives, done_collisions = env.step([7])
        print(f'observations: {observations}')
        print(f'rewards: {rewards}')
        print(f'done_arrives: {done_arrives}')
        print(f'done_collisions: {done_collisions}')
        print('-------------------------------------------------')
        if done_arrives[0] or done_collisions[0]:
            break
