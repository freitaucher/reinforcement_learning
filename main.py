import numpy as np
from math import prod
import random
import cv2
import shutil,os
import json
from copy import deepcopy
from utils import init_qtable, init_image, init_environment, reward, index2lin, do_step
from plot_qtable import plot_qtable


def starting_step(config):

    if eval(config["environment"]["new"]):
        print('\nStart from scratch!\n')
        
    env_shape = tuple(config["environment"]["shape"])
    if not os.path.exists(config["environment"]["saved"]) or eval(config["environment"]["new"]):
        init_environment(env_shape=env_shape, danger_ratio=config["environment"]["danger_ratio"], stop_len=config["environment"]["number_of_exits"])
        env_data = np.load(config["environment"]["saved"],allow_pickle=True)
        env,stop,danger,indices_free = env_data['env'],env_data['stop'],env_data['danger'],env_data['indices']       
        print('environment is initialized...')
        img0 = init_image(env,stop,danger,res=config["resolution"])   
        cv2.imwrite(config['environment']['img'],img0)

    img0 = cv2.imread(config['environment']['img'])
        
    env_data = np.load(config["environment"]["saved"],allow_pickle=True)
    env,stop,danger,indices_free = env_data['env'],env_data['stop'],env_data['danger'],env_data['indices']
    if env.shape !=  env_shape:
        print('the requested in config.json environment shape', env.shape, 'differs from one loaded from', config['environment']['saved'],':',env_shape)
        quit()    
   
    if not os.path.exists(config["qtable_last"]) or eval(config["environment"]["new"]):
        print('qtable is initialized...')
        init_qtable(env)
        
    qtable = np.load(config["qtable_last"])
    if qtable.shape[0] != env_shape[0] * env_shape[1] or qtable.shape[1] != 4:
        print('loaded qtable shape',qtable.shape,' is inconsistent with environment:',env_shape)
        quit()
          
    print('environment shape:',env.shape, 'stop points shape:', stop.shape, 'trap points shape:', danger.shape)    
    
    return env,stop,danger,indices_free, img0, qtable





if __name__ == "__main__":
  
    
    f = open("config.json")
    config =  json.load(f)
    f.close()    
    outdir = config["outdir"]    
    shutil.rmtree(outdir,ignore_errors=True)
    os.mkdir(outdir)    
    res = config["resolution"]    
    n_episodes  = config["n_episodes"]
    env,stop,danger,indices_free, img0, qtable = starting_step(config)
    count_max = prod(env.shape)

    
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
            s, qtable, reward_val, qtable_updated = do_step(s_old, env, qtable, random_step_prob=config["random_step_prob"], lr=config["learning_rate"], gamma=config["gamma"], qtable_save=config["qtable_last"])

            #ds = model(s_old)
            #s = s_old + ds
            #loss =  (1-lr) * reward(s)  + lr * gamma * np.min(np.array([reward(s+ds) for ds in steps]))
            
            ########################################################################################################
            rewards.append(reward_val)            
            #print('do_step',count,'from', s_old, 'to', s, 'end:',stop)

            if qtable_updated:
                print('qtable is updated!')
                img0 = plot_qtable(config['environment']['saved'], config['qtable_last'], res=(res,res))
            
            # redraw:
            img = deepcopy(img0) # wipe start
            img[s[0]*res+res//4:s[0]*res+(3*res)//4, s[1]*res+res//4:s[1]*res+(3*res)//4] = (100,0,0) # set start
            
            if episode % config["save_every"] == 0:
                cv2.imwrite(outdir+'/'+str(episode).zfill(6)+'/'+str(count+1).zfill(4)+'.png',img)
                
            count+=1
            
            if env[s[0], s[1], s[2]] == -1:
                print('agent is killed!')
                break

            if count > count_max :
                #print('agent is arrested!')
                break

        print('episode:  %8d' % episode, 'mean reward: %10.3f' % np.mean(rewards), 'number of steps: %4d' % count)
        

