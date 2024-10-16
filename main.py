import numpy as np
import random
from math import prod
from copy import deepcopy
import cv2
import shutil,os
import json


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



def init_environment(env_shape=(15,15,15),alpha=-1, stop_len=1):
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
        
    
    danger_len = random.randint(2,len(indices)//4)
    danger=[]
    for i in range(danger_len):        
        danger.append( random.choice(list(indices)) )
        indices = indices - set([danger[-1]])
        env[danger[-1]] = alpha
        
    
    # reshape indices:
    env_reshaped = np.array(env,dtype=int).reshape(env_shape)
    #start_reshaped = np.array([ start//(env_shape[1]*env_shape[2]), (start % (env_shape[1]*env_shape[2])) // env_shape[2], start %  env_shape[2]], dtype=int)
    #stop_reshaped = np.array([ stop//(env_shape[1]*env_shape[2]), (stop % (env_shape[1]*env_shape[2])) // env_shape[2], stop %  env_shape[2]], dtype=int)
    danger_reshaped = []
    for i in range(danger_len):
        danger_reshaped.append(  [danger[i]//(env_shape[1]*env_shape[2]), (danger[i] % (env_shape[1]*env_shape[2])) // env_shape[2], danger[i] %  env_shape[2]] )
        
    danger_reshaped = np.array(danger_reshaped,dtype=int)

    
    np.savez('env.npz', env=env_reshaped, stop=stop_reshaped, danger=danger_reshaped, indices=indices)

    return None
    #return env_reshaped, stop_reshaped, danger_reshaped, indices



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



def do_step(s, env, qtable, eps=1, lr=0.1, gamma=0.1):

    steps = [[-1,0,0],[1,0,0],[0,-1,0],[0,1,0]]
    order = np.argsort( -qtable[index2lin(s,env)])


    if eps<0:
        eps = random.uniform(0,1)
    

    if eps > 0.5:    
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
    
    
    if qtable[index2lin(s,env),step_index] != old:
        print('qtable is updated!')
        np.save('qlast.npy', qtable)

    
    return s_new, qtable, reward_value



##########################################################################################

if __name__ == "__main__":

    f = open("config.json")
    config =  json.load(f)
    f.close()
    
    outdir = config["outdir"]    
    shutil.rmtree(outdir,ignore_errors=True)
    os.mkdir(outdir)
    
    res = config["resolution"]
    
    n_episodes  = config["n_episodes"]

    if not os.path.exists(config["environment_saved"]):
        env_shape = tuple(config["environment_shape"])
        init_environment(env_shape=env_shape, stop_len=config["environment_number_of_exits"])

    env_data = np.load(config["environment_saved"],allow_pickle=True)

    env,stop,danger,indices_free = env_data['env'],env_data['stop'],env_data['danger'],env_data['indices']
   
    if not os.path.exists(config["qtable_last"]):
        init_qtable(env)
    qtable = np.load(config["qtable_last"])
          
    print('environment:',env.shape, stop.shape, danger.shape)    
    img0 = init_image(env,stop,danger,res=res)   
    cv2.imwrite(config['environment_img'],img0)

   

    
    for episode in range(n_episodes):

        rewards=[]
        
        shutil.rmtree(outdir+'/'+str(episode).zfill(6),ignore_errors=True)
        if episode % config["save_every"] == 0:
            os.mkdir(outdir+'/'+str(episode).zfill(6))

        start = random.choice(list(indices_free.item()))
        start_reshaped = np.array([ start//(env.shape[1]*env.shape[2]), (start % (env.shape[1]*env.shape[2])) // env.shape[2], start %  env.shape[2]], dtype=int)
        start0 = deepcopy(start_reshaped)

        img = deepcopy(img0)
        img[start0[0]*res+res//4:start0[0]*res+(3*res)//4, start0[1]*res+res//4:start0[1]*res+(3*res)//4] = (100,0,0) # set start
        if episode % config["save_every"] == 0:
            cv2.imwrite(outdir+'/'+str(episode).zfill(6)+'/'+str(0).zfill(4)+'.png',img)
        
        s = start_reshaped
        count=0
       
        while env[s[0], s[1], s[2]] != 999:
            s_old =  s
            ########################################################################################################            
            s, qtable, reward_val = do_step(s_old, env, qtable, eps=config["eps"], lr=config["learning_rate"], gamma=config["gamma"])
            ########################################################################################################
            rewards.append(reward_val)            
            print('do_step',count,'from', s_old, 'to', s, 'end:',stop)

            # redraw:
            img = deepcopy(img0) # wipe start
            img[s[0]*res+res//4:s[0]*res+(3*res)//4, s[1]*res+res//4:s[1]*res+(3*res)//4] = (100,0,0) # set start
            
            if episode % config["save_every"] == 0:
                cv2.imwrite(outdir+'/'+str(episode).zfill(6)+'/'+str(count+1).zfill(4)+'.png',img)
                
            count+=1
            
            if env[s[0], s[1], s[2]] == -1:
                print('agent is killed!')
                break

            if count > prod(env.shape):
                print('agent is arrested!')
                break

        print('episode:  %8d' % episode, 'mean reward: %10.3f' % np.mean(rewards), 'number of steps: %4d' % count)
        

