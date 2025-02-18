import DDQNAgent
import numpy as np
import YpacaraiMap
from torch import load
import matplotlib.pyplot as plt

# 初始化环境
env = YpacaraiMap.Environment()

# 设置与训练时相同的参数
gamma = 0.95
epsilon = 0.0  # 测试时不需要探索，将epsilon设为0
lr = 1e-3
n_actions = 8
input_dims = (3,25,19)
mem_size = 10000
batch_size = 250
eps_min = 0.01
eps_dec = 0
replace = 50

# 创建agent并加载训练好的模型
agent = DDQNAgent.DDQNAgent(gamma, epsilon, lr, n_actions, input_dims, 
                           mem_size, batch_size, eps_min, eps_dec, replace)
agent.q_eval.load_state_dict(load("DDQN_BEST.pt"))  # 加载最佳模型
agent.q_eval.eval()  # 设置为评估模式

def do_step(env, action):
    obs, reward, done, info = env.step(action)       
    state = env.render()   
    return state, reward, done, info
    
def reset(env):
    env.reset()    
    state = env.render()
    return state

# 测试模型
n_episodes = 5  # 测试5个回合
steps = 300     # 每个回合的最大步数

for episode in range(n_episodes):
    state = reset(env)
    total_reward = 0
    
    print(f"\n开始第 {episode+1} 个回合")
    
    for step in range(steps):
        # 选择动作（使用训练好的模型）
        action = agent.choose_action_epsilon_greedy(state)
        
        # 执行动作
        next_state, reward, done, info = do_step(env, action)
        
        total_reward += reward
        state = next_state
        
        # 可以添加可视化或其他信息打印
        print(f"Step {step+1}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
        
        if done:
            print(f"回合结束于步数 {step+1}")
            break
    
    print(f"回合 {episode+1} 总奖励: {total_reward:.2f}")

print("\n测试完成!") 