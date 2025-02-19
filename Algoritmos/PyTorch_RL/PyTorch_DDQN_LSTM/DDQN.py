#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import DDQNAgent
import numpy as np
import matplotlib.pyplot as plt
import YpacaraiMap
from torch import save
from tqdm import tqdm

steps = 300
epochs = 1500
gamma = 0.95
epsilon = 0.99
lr = 1e-3
n_actions = 8
mem_size = 10000
batch_size = 250
eps_min = 0.01
eps_dec = (epsilon-eps_min)/1400
replace = 50
timesteps = 5
input_dims = (timesteps,3,25,19)

agente = DDQNAgent.DDQNAgent(gamma, epsilon, lr, n_actions, input_dims, 
                 mem_size, batch_size, eps_min, eps_dec, replace)

env = YpacaraiMap.Environment()

def do_step(env,action,ext_state):    
    obs, reward, done, info = env.step(action)       
    state = env.render()    
    for t in range(timesteps-1):
        ext_state[t] = ext_state[t+1]        
    ext_state[timesteps-1] = state    
    return ext_state, reward, done, info
    
def reset(env):    
    env.reset()       
    state = env.render()    
    ext_state = np.zeros((timesteps,3,25,19))    
    for t in range(timesteps):        
        ext_state[t] = state     
    return ext_state

np.random.seed(42)

filtered_reward = 0
filtered_reward_buffer = []
reward_buffer = []

record = -1000

for epoch in tqdm(range(0,epochs)):    
    state = reset(env)
    rew_episode = 0
    agente.decrement_epsilon()
    
    for step in range(steps):
        action = agente.choose_action_epsilon_greedy(state)
        next_state, reward, done, info = do_step(env,action,state)
        
        agente.store_transition(state,action,reward,next_state,done)
        state = next_state
        
        rew_episode += reward
        agente.learn()
        
    agente.replace_target_network(epoch)

    if epoch == 0:
        filtered_reward = rew_episode
    else:
        filtered_reward = rew_episode*0.05 + filtered_reward*0.95
    
    reward_buffer.append(rew_episode)
    filtered_reward_buffer.append(filtered_reward)

    if(record < rew_episode):
        print(f"New Record: {rew_episode:06.2f} at Episode {epoch:d}\n")
        record = rew_episode
        save(agente.q_eval, "DDQN_BEST.pt")

print('end!')

plt.figure(figsize=(8, 4))
plt.plot(reward_buffer, 'b', alpha=0.2, label='Raw Reward')
plt.plot(filtered_reward_buffer, 'r', label='Filtered Reward')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Training Rewards Over Time')
plt.grid(True, which='both')
plt.legend()
plt.show()

save(agente.q_eval, "DDQN_LAST.pt")
        
        

