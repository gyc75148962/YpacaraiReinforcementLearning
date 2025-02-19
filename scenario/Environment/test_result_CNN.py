#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from time import sleep
import pushover
import random
import os
from IPython.display import clear_output
from collections import deque
import progressbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Bibliotecas de NNs (Keras y Tensorflow)
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape,Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import YpacaraiMap

env = YpacaraiMap.Environment()
env.set_test_mode(True)

fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,8))

ax2.set_xticks(np.arange(env.S.shape[1]))
ax2.set_yticks(np.arange(env.S.shape[0]))
ax2.grid(True, linewidth = 0.5, alpha = 0.1, drawstyle = 'steps-mid')
plt.setp(ax2.get_xticklabels(), rotation_mode="anchor")


ax3.set_xticks(np.arange(env.S.shape[1]))
ax3.set_yticks(np.arange(env.S.shape[0]))
ax3.grid(True, linewidth = 0.5, alpha = 0.2, drawstyle = 'steps-mid')

plt.setp(ax3.get_xticklabels(), rotation_mode="anchor")

fig.suptitle('Ship status map')


clear_session()


def do_step(env,action):
    
    obs, reward, done, info = env.step(action)
       
    state = np.dstack((obs['visited_map'],obs['importance_map']))
    
    return state, reward, done, info
    
def reset(env):
    
    obs = env.reset()
       
    state = np.dstack((obs['visited_map'],obs['importance_map']))
    
    return state    

obs = env.reset()

state = env.render()

N = 500
grid_map = env.map
position = obs['position']

model = keras.models.load_model('./DQN2D_Ypacarai_Model_BEST.h5')
reward = 0
num = 0

for steps in range(N):
   
    q_values = model.predict(state[np.newaxis])
    
    if np.random.rand()<0.9:
        action = np.argmax(q_values[0])
    else:
        action = np.random.randint(0,8)
        
    valid = 0
    while valid == 0:
        
        if action == 0: # NORTH
            if grid_map[position[0]-1][position[1]] == 0:
                valid = 0
            else:
                valid = 1
                
        elif action == 1: # SOUTH
            if grid_map[position[0]+1][position[1]] == 0:
                valid = 0
            else:
                valid = 1
                
        elif action == 2: # EAST
            if grid_map[position[0]][position[1]+1] == 0:
                valid = 0
            else:
                valid = 1
                
        elif action == 3: # WEST
            if grid_map[position[0]][position[1]-1] == 0:
                valid = 0
            else:
                valid = 1
                
        elif action == 4: # NE
            if grid_map[position[0]-1][position[1]+1] == 0:
                valid = 0
            else:
                valid = 1
                
        elif action == 5: # NW
            if grid_map[position[0]-1][position[1]-1] == 0:
                valid = 0
            else:
                valid = 1
                
        elif action == 6: # SE
            if grid_map[position[0]+1][position[1]+1] == 0:
                valid = 0
            else:
                valid = 1
                
        elif action == 7: # SW
            if grid_map[position[0]+1][position[1]-1] == 0:
                valid = 0
            else:
                valid = 1
                
        if valid == 0:
            action = np.random.randint(0,8)
            
            
    obs,rew,done,info = env.step(action)
    
    print(f"Step n: {steps}")
    
    reward += rew
   
    state = env.render()
    position = obs['position']
        
    print(f"The reward for this action has been: {rew:.3f}")

print(f"Finished with reward {reward}")

VM = obs['visited_map']
IM = obs['importance_map']
img = env.render()

ax1.imshow(img)
ax2.imshow(VM, cmap = 'gray')
im = ax3.imshow(IM,interpolation='bicubic', cmap = 'jet_r')
divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

plt.show()

metrics = env.metrics()


print("Coverage: {}".format(metrics['coverage']))
print("Media de tiempo: {}".format(metrics['mean']))
print("Dev. tipica: {}".format(metrics['std']))

fig = plt.figure()

plt.imshow(metrics['time_matrix'])
plt.colorbar()

plt.show()
    
