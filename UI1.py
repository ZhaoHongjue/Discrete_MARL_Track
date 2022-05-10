import numpy as np
import tkinter as tk

UNIT = 40

class Maze(tk.Tk, object):
    '''
    环境的图形化界面
    '''
    def __init__(self, map):
        '''
        相关的初始化
        '''
        super(Maze, self).__init__()
        # 初始化地图规模
        self.map = map
        self.MAZE_W = self.map.shape[0]
        self.MAZE_H = self.map.shape[1]

        self.title('maze')
        self.geometry('{0}x{1}'.format(self.MAZE_H * UNIT, self.MAZE_H * UNIT))
        self._build_maze()
    
    def _build_maze(self):
        '''
        创建画布，绘制图形化界面
        '''
        self.canvas = tk.Canvas(self, bg='white',
                           height=self.MAZE_H * UNIT,
                           width=self.MAZE_W * UNIT)

        # 创建网格
        for c in range(0, self.MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 创建坐标
        origin = np.array([20, 20])

        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i, j] == 1: # 障碍物采用黑色方块表示
                    hell_center = origin + np.array([UNIT * i, UNIT * j])
                    self.canvas.create_rectangle(
                        hell_center[0] - 20, hell_center[1] - 20,
                        hell_center[0] + 20, hell_center[1] + 20,
                        fill='black')
                elif self.map[i, j] == 2: # 我方机器人采用黄色点表示
                    hunter_center = origin + np.array([UNIT * i, UNIT * j])
                    self.canvas.create_oval(
                        hunter_center[0] - 15, hunter_center[1] - 15,
                        hunter_center[0] + 15, hunter_center[1] + 15,
                        fill='yellow')
                elif self.map[i, j] == 3: # 目标点采用绿色方块表示
                    goal_center = origin + np.array([UNIT * i, UNIT * j])
                    self.canvas.create_rectangle(
                        goal_center[0] - 20, goal_center[1] - 20,
                        goal_center[0] + 20, goal_center[1] + 20,
                        fill='green')
                elif self.map[i, j] == 4: # 敌方机器人采用红色点表示
                    prey_center = origin + np.array([UNIT * i, UNIT * j])
                    self.canvas.create_oval(
                        prey_center[0] - 15, prey_center[1] - 15,
                        prey_center[0] + 15, prey_center[1] + 15,
                        fill='red')
        # pack all
        self.canvas.pack()
    
if __name__ == "__main__":
    test = np.zeros((20, 20))
    test[1, 0] = 1
    test[4, 0] = 2
    test[7, 0] = 3
    test[10, 0] = 4
    env = Maze(test)
    env.mainloop()