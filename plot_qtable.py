import numpy as np
import cv2
import os, sys
import matplotlib.colors as mcolors
from utils import init_image
from math import prod

def gray8rgb8(img, color_levels=["maroon", "darkred", "white", "deepskyblue", "blue"]):
    #
    # color_levels: plain list of colors used for mapping grayshades from 0 to 1
    # img: gray8 image on input; rgb8 on output
    #
    cmap = mcolors.LinearSegmentedColormap.from_list("", color_levels, N=2**8)           
    img  = img/(2**8-1)
    img = (cmap(img) * (2**8-1)).astype(np.uint8)[:,:,:3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
    return img



def value2color(x):
    #return int((x+1)*127.5)
    #return int((x+1)*(x+3)*255/4)
    return int( 255 / (1+np.exp(-x))) 

#  steps = [[-1,0,0],[1,0,0],[0,-1,0],[0,1,0]]
#  order = np.argsort( -qtable[index2lin(s,env)])
#

def draw_arrow(image, x,y, step_index, res, qval=0):
    cy,cx=int((y+0.5)*res[0]),int((x+0.5)*res[1])

    start_point, end_point = (0,0), (0,0)  
    
    if step_index == 3:
        start_point = (int(cy-0.25*res[0]),cx)
        end_point =   (int(cy+0.25*res[0]),cx)
    if step_index == 2:
        start_point = (int(cy+0.25*res[0]),cx)
        end_point =   (int(cy-0.25*res[0]),cx)
    if step_index == 1:
        start_point = (cy, int(cx-0.25*res[1]))
        end_point =   (cy, int(cx+0.25*res[1])) 
    if step_index == 0:
        start_point = (cy, int(cx+0.25*res[1]))
        end_point =   (cy, int(cx-0.25*res[1]))


    c = 255 - value2color(qval)
    color=(c,c,c)
    thickness = int( 5 / (1+np.exp(-qval))) 
    
    image = cv2.arrowedLine(image, start_point, end_point, color=color, thickness=thickness, tipLength = 0.25)

    return image



fname=str(sys.argv[1])
env_data = np.load(fname,allow_pickle=True)
env,stop,danger,indices_free = env_data['env'],env_data['stop'],env_data['danger'],str(env_data['indices'])[1:-1]
indices_free = [int(i) for i in indices_free.split(',')]
print(env.shape, len(danger), len(stop), len(indices_free), (len(danger)+len(stop)+len(indices_free)),'=',prod(env.shape))
#quit()
fname=str(sys.argv[2])
qtable=np.load(fname)
print(qtable.shape, 'min, max:',np.min(qtable),np.max(qtable))

res = (50, 50)
img = init_image(env, stop, danger, res=res[0])


for i,ql in enumerate(qtable):
    x = i // env.shape[0]
    y = i % env.shape[0]
    if i in indices_free:
        #if len(set(ql)) > 1:
        order = np.argsort(-ql)
        #"""
        step_index=order[0]
        img = draw_arrow(img, x,y, step_index, res, ql[step_index])
        print(i, x, y, ql, step_index)
        """
        for step_index in order[::-1]:
        img = draw_arrow(img, x,y, step_index, res, ql[step_index])
        print(i, x, y, ql, step_index)
        """

cv2.imwrite(fname[:-3]+'png',img)



