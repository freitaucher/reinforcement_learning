import numpy as np
from math import prod
import random
from copy import deepcopy


def init_qtable(env):
    qtable = np.zeros((prod(env.shape), 4))
    np.save('qlast.npy', qtable)    
    return None


def init_image(env,stop,danger, res=50):
    img = np.ones( (env.shape[0] * res, env.shape[1] * res, 3) ) * 255 
    for i in range(env.shape[0]):        
        img[i*res,:] = (0, 0, 0)       
    for j in range(env.shape[1]):
        img[:,j*res] = (0, 0, 0)

    stop = stop * res
    danger = danger * res

    for s in stop:
        img[s[0]:s[0]+res, s[1]:s[1]+res] = (0,100,0)

    for d in danger:
        img[d[0]:d[0]+res,d[1]:d[1]+res] = (0,0,100)
    
    img = np.array(img).astype(np.uint8)
    
    return img



def init_environment(env_shape=(15,15,15), danger_ratio=0.2,  alpha=-1, stop_len=1):
    env = [0] * prod(env_shape)
    indices = set([i for i in range(len(env))])

    stop = []
    for i in range(stop_len):
        stop.append( random.choice(list(indices)) )
        env[stop[i]] = 999 
        indices = indices - set([stop[-1]])
        
    stop_reshaped = []
    for i in range(stop_len):
        stop_reshaped.append(  [stop[i]//(env_shape[1]*env_shape[2]), (stop[i] % (env_shape[1]*env_shape[2])) // env_shape[2], stop[i] %  env_shape[2]] )
        
    
    danger_len = int(len(indices)*danger_ratio) #random.randint(2,len(indices)//4)
    
    danger=[]
    for i in range(danger_len):        
        danger.append( random.choice(list(indices)) )
        indices = indices - set([danger[-1]])
        env[danger[-1]] = alpha        
    
    # reshape indices:
    env_reshaped = np.array(env,dtype=int).reshape(env_shape)
    danger_reshaped = []
    for i in range(danger_len):
        danger_reshaped.append(  [danger[i]//(env_shape[1]*env_shape[2]), (danger[i] % (env_shape[1]*env_shape[2])) // env_shape[2], danger[i] %  env_shape[2]] )
        
    danger_reshaped = np.array(danger_reshaped,dtype=int)
    
    np.savez('env.npz', env=env_reshaped, stop=stop_reshaped, danger=danger_reshaped, indices=indices)

    return None



def reward(s, env):
    reward_val=0
    if env[s[0],s[1],s[2]]==999:
        reward_val=1
    if env[s[0],s[1],s[2]]==-1:
        reward_val=-1        
    return reward_val



def index2lin(s, env):
    index_lin=s[0]*(env.shape[0]*env.shape[2])+s[1]*env.shape[2]+s[2]
    return index_lin



def do_step(s, env, qtable, random_step_prob=1, lr=0.1, gamma=0.1, qtable_save='qlast.npy'):

    steps = [[-1,0,0],[1,0,0],[0,-1,0],[0,1,0]]
    order = np.argsort( -qtable[index2lin(s,env)])
    
    eps = random.uniform(0,1)    

    if eps > random_step_prob:    
        for step_index in order:
            s_new  = s + steps[step_index] # take the best according to qtable
            allowed = True
            for dim in [0,1,2]:
                if s_new[dim]  < 0 or s_new[dim] > env.shape[dim]-1:
                    allowed=False
            if  s_new[0]==s[0] and s_new[1]==s[1] and s_new[2]==s[2]:
                allowed = False
            if allowed:
                break
    else:
        allowed = False
        while not allowed:        
            step_index = random.choice([i for i in range(len(steps))])  # take random
            s_new  = s + steps[step_index]
            allowed = True
            for dim in [0,1,2]:
                if s_new[dim]  < 0 or s_new[dim] > env.shape[dim]-1:
                    allowed=False
            if  s_new[0]==s[0] and s_new[1]==s[1] and s_new[2]==s[2]:
                allowed = False
            if allowed:
                break        

        
    order_new = np.argsort(-qtable[index2lin(s_new,env)])
    for step_index_new in order_new:
        s_new_new  = s_new + steps[step_index_new]
        allowed_new = True
        for dim in [0,1,2]:
            if s_new_new[dim]  < 0 or s_new_new[dim] > env.shape[dim]-1:
                allowed_new=False
        if  s_new_new[0]==s_new[0] and s_new_new[1]==s_new[1] and s_new_new[2]==s_new[2]:
            allowed_new = False
        if allowed_new:
            break
        

    old = deepcopy(qtable[index2lin(s,env),step_index])
    reward_value = reward(s_new,env)
    qtable[index2lin(s,env),step_index] = old  + lr * ( reward_value  + gamma * qtable[index2lin(s_new,env), step_index_new] - old )
    

    qtable_updated=False
    if qtable[index2lin(s,env),step_index] != old:
        np.save(qtable_save, qtable)
        qtable_updated=True

    
    return s_new, qtable, reward_value, qtable_updated
