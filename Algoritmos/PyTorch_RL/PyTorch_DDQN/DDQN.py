#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import DDQNAgent
import numpy as np
import matplotlib.pyplot as plt
import YpacaraiMap
from torch import save
from tqdm import tqdm

steps = 300
epochs = 2000
gamma = 0.95
epsilon = 0.99
lr = 1e-3
n_actions = 8
input_dims = (3,25,19)
mem_size = 10000
batch_size = 250
eps_min = 0.01
eps_dec = (epsilon-eps_min)/1400
replace = 20

agent = DDQNAgent.DDQNAgent(gamma, epsilon, lr, n_actions, input_dims, 
                 mem_size, batch_size, eps_min, eps_dec, replace)
env = YpacaraiMap.Environment()

def do_step(env,action):
    obs, reward, done, info = env.step(action)       
    state = env.render()   
    return state, reward, done, info
    
def reset(env):
    env.reset()    
    state = env.render()
    return state

np.random.seed(42)

filtered_reward = 0
filtered_reward_buffer = []
reward_buffer = []

record = -1000

for epoch in tqdm(range(0,epochs)):    
    state = reset(env)
    rew_episode = 0
    agent.decrement_epsilon()
    
    for step in range(steps):
        action = agent.choose_action_epsilon_greedy(state)
        next_state, reward, done, info = do_step(env,action)
        
        agent.store_transition(state,action,reward,next_state,done)
        
        state = next_state        
        rew_episode += reward
        
        agent.learn()
        
    # 更新目标网络
    agent.replace_target_network(epoch)

    if epoch == 0:
        filtered_reward = rew_episode
    else:
        filtered_reward = rew_episode*0.05 + filtered_reward*0.95
    
    reward_buffer.append(rew_episode)
    filtered_reward_buffer.append(filtered_reward)

    if(record < rew_episode):
        print(f"New Record: {rew_episode:06.2f} at Episode {epoch:d}\n")
        record = rew_episode
        save(agent.q_eval, "DDQN_BEST.pt")

print('end!')

save(agent.q_eval, "DDQN_LAST.pt")

# 训练结束后统一绘图
plt.figure(figsize=(8, 4))
plt.plot(reward_buffer, 'b', alpha=0.2, label='Original Rewards') 
plt.plot(filtered_reward_buffer, 'r', label='Filtered Rewards')    
plt.xlabel('Episodes')   
plt.ylabel('Rewards')   
plt.title('Training Rewards over Time') 
plt.grid(True, which='both')
plt.legend()
plt.xlim([0, epochs])

# 保存图片
plt.savefig(f'training_rewards-{epochs}.png', dpi=300, bbox_inches='tight')
plt.close()
        
        

