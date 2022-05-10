from importlib.resources import path
import numpy as np
from PIL import Image, ImageDraw

k = 20 
pixel = 30
champ = 3

class drawmap:
    
    def __init__(self,size,pixel,map):
        '''
        size+2为边长的白色格子图
        '''
        self.map = map
        self.size=size
        self.pixel = pixel
        self.image = Image.new('RGBA',[(size+2)*pixel,(size+2)*pixel],color =  "white")
    
    def draw_grid(self):
        '''
        划线,size+1条
        '''
        size = (self.size)
        pixel = self.pixel
        draw = ImageDraw.Draw(self.image)
        for i in range (1,size+2):
            draw.line([(i*pixel,(size+1)*pixel),(i*pixel,pixel)],fill = (0,0,0,255),width = 2)
            draw.line([((size+1)*pixel,i*pixel),(pixel,i*pixel)],fill = (0,0,0,255),width = 2)
        draw.line([((size+1)*pixel,pixel),(pixel,pixel)],fill = (0,0,0,255),width =5)
        draw.line([(pixel,(size+1)*pixel),(pixel,pixel)],fill = (0,0,0,255),width =5)
        draw.line([(pixel,(size+1)*pixel),((size+1)*pixel,(size+1)*pixel)],fill = (0,0,0,255),width =5)
        draw.line([((size+1)*pixel,pixel),((size+1)*pixel,(size+1)*pixel)],fill = (0,0,0,255),width =5)

    def draw_case(self,i, j, idx):
        '''
        画机器人与猎物
        '''
        pixel = self.pixel
        draw = ImageDraw.Draw(self.image)
        if idx == 1:
            draw.rectangle([((i+1)*pixel,(j+1)*pixel),((i+2)*pixel,(j+2)*pixel)],fill = 'black')
        elif idx == 2:
            draw.ellipse([((i+1)*pixel,(j+1)*pixel),((i+2)*pixel,(j+2)*pixel)],fill = 'red')
        elif idx == 3:
            draw.rectangle([((i+1)*pixel,(j+1)*pixel),((i+2)*pixel,(j+2)*pixel)],fill = 'grey')
        elif idx == 4:
            draw.ellipse([((i+1)*pixel,(j+1)*pixel),((i+2)*pixel,(j+2)*pixel)],fill = 'green')
    

    def show(self):
        self.image.show()
    
    def draw_map(self):
        '''
        作图并保存
        '''
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                self.draw_case(i, j, self.map[i, j])
        self.draw_grid()
        return self.image

def show_video(images, n) :
    '''
    做成gif
    '''
    title = "result" + str(n) +".gif"
    images[0].save(title,save_all=True, append_images=images[1:],duration=100, loop=False, optimize=True, path = './gif')
    
if __name__ == '__main__':
    # Map = [np.zeros((20, 20))] * 20
    # Test = drawmap(20, pixel=30, map = Map)
    # show_video(Test)
    image = Image.new('RGBA',[(20+2)*pixel,(20+2)*pixel],color =  "white")
    image.save('fuck.png')