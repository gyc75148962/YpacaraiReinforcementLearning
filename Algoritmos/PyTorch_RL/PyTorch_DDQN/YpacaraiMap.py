#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import genfromtxt

class Environment:    
    def __init__(self):        
        # 加载地图并创建状态矩阵#
        self.map = genfromtxt('YpacaraiMap.csv', delimiter=',', dtype=int)
        self.S = np.ones(self.map.shape)
        self.R_abs = genfromtxt('YpacaraiMap_variable.csv', delimiter=',', dtype=int)
        self.visited = np.zeros(self.map.shape)
        
        # 用于指标的映射#
        self.visited_M = np.zeros(shape = self.map.shape)
        self.waiting_M = np.ones(shape = self.map.shape)
        self.mean_time_M = np.zeros(shape = self.map.shape)
        
        # 在训练中执行额外指标可能会减慢训练速度 #
        self.test_mode = False        
        # 选择一个随机起点 #
        posible_x, posible_y = np.nonzero(self.map)       
        init_cell_index = np.random.randint(0,posible_x.size)        
        self.agent_start_pos = (posible_x[init_cell_index],posible_y[init_cell_index])

        self.agent_pos = self.agent_start_pos
        self.agent_pos_ant = self.agent_start_pos
        
        # 将起点标记为已访问 #
        self.visited[self.agent_pos[0]][self.agent_pos[1]] = 255
        self.S[self.agent_pos[0]][self.agent_pos[1]] = 0
        #self.R_abs[self.agent_pos[0]][self.agent_pos[1]] -= 50
        
    def reset(self):
        self.S = np.ones(self.map.shape)
        self.R_abs = genfromtxt('YpacaraiMap_variable.csv', delimiter=',', dtype=int)
        self.visited = np.zeros(self.map.shape)
        
        self.visited_M = np.zeros(shape = self.map.shape)
        self.waiting_M = np.ones(shape = self.map.shape)
        self.mean_time_M = np.zeros(shape = self.map.shape)
        
        # 选择一个随机起点 #
        posible_x, posible_y = np.nonzero(self.map)
        
        init_cell_index = np.random.randint(0,posible_x.size)
        
        self.agent_start_pos = [posible_x[init_cell_index],posible_y[init_cell_index]]

        self.agent_pos = self.agent_start_pos
        self.agent_pos_ant = self.agent_start_pos
        
        # 将起点标记为已访问 #
        self.visited[self.agent_pos[0]][self.agent_pos[1]] = 255
        self.S[self.agent_pos[0]][self.agent_pos[1]] = 0
        
        # 计算状态 #
        obs = {}
        obs['visited_map'] = self.visited
        obs['importance_map'] = self.S*self.R_abs
        obs['position'] = self.agent_pos
        
        return obs
        
    def step(self,action):
        future_pos = np.copy(self.agent_pos_ant)        
        if action == 0: # North
             future_pos[0] -= 1
        elif action == 1: # South
             future_pos[0] += 1
        elif action == 2: # East #
             future_pos[1] += 1
        elif action == 3: # West #
             future_pos[1] -= 1
        elif action == 4: # NE #
             future_pos[0] -= 1
             future_pos[1] += 1
        elif action == 5: # NW #
             future_pos[0] -= 1
             future_pos[1] -= 1
        elif action == 6: # SE #
             future_pos[0] += 1
             future_pos[1] += 1
        elif action == 7: # SW #
             future_pos[0] += 1
             future_pos[1] -= 1
            
        # 检查是否是非法移动 #
        ilegal = 0
        if self.map[future_pos[0]][future_pos[1]] == 0:
            ilegal = 1
        else:
            self.agent_pos = future_pos
            
        # 前一个位置标记在访问地图上 #
        self.visited[self.agent_pos_ant[0]][self.agent_pos_ant[1]] = 127 # 前一个位置阴影 
        self.visited[self.agent_pos[0]][self.agent_pos[1]] = 255 # 已访问位置标记良好
        
        # 清除前一个位置的兴趣 #
        self.R_abs[self.agent_pos_ant[0]][self.agent_pos_ant[1]] -= 50 
        
        # 处理奖励 #
        rho_next = self.R_abs[self.agent_pos[0]][self.agent_pos[1]] * self.S[self.agent_pos[0]][self.agent_pos[1]]
        rho_act = self.R_abs[self.agent_pos_ant[0]][self.agent_pos_ant[1]] * self.S[self.agent_pos_ant[0]][self.agent_pos_ant[1]]
        
        # 计算奖励作为兴趣梯度 #
        reward = rho_next - rho_act
        # 注意这里我们决定惩罚访问非法、前一个位置或新位置的程度 #
        reward = (1-ilegal)*((5.505/255)*(reward-255)+5) - ilegal*(10)
        
        # 更新矩阵S #
        for i in range(0,self.map.shape[0]):
            for j in range(0,self.map.shape[1]):
                self.S[i][j] = np.min([self.S[i][j]+0.05, 1])
        self.S[self.agent_pos[0]][self.agent_pos[1]] = 0.1
        
        self.agent_pos_ant = self.agent_pos
        
        obs = {}
        obs['visited_map'] = self.visited
        obs['importance_map'] = self.S*self.R_abs
        obs['position'] = self.agent_pos
        
        done = 0
        
        # 对于测试模式激活 #
        if(self.test_mode == True):
            # 增加对应单元格的访问次数 #
            self.visited_M[self.agent_pos[0]][self.agent_pos[1]] += 1
            # 获取访问该单元格所花费的时间 #
            T = self.waiting_M[self.agent_pos[0]][self.agent_pos[1]]
            # 增加除访问单元格外的所有单元格的等待时间 #
            self.waiting_M +=  1
            self.waiting_M[self.agent_pos[0]][self.agent_pos[1]] = 1
            # 累积该单元格的访问时间 #
            self.mean_time_M[self.agent_pos[0]][self.agent_pos[1]] += T
            
        else:
            pass
        return obs, reward, done, ilegal
    
    def render(self):
        green_color = np.array([0,160,20])/255
        blue_color = np.array([0,0,0])/255
        agent_color = np.array([255,0,0])/255
        red_color = np.array([241,241,241])/255
        
        # 制作地图的副本 #
        size_map = (3,self.map.shape[0],self.map.shape[1])
        base_map = np.zeros(size_map)
        
        for i in range(0,self.map.shape[0]):
            for j in range(0,self.map.shape[1]):
                if(self.map[i][j] == 0):#无法访问
                    base_map[:,i,j] = green_color
                elif(self.agent_pos[0] == i and self.agent_pos[1] == j):#当前位置
                    base_map[:,i,j] = agent_color
                elif(self.visited[i][j] != 0):#被访问过
                    state_of_visited = (255-self.R_abs[i][j]*self.S[i][j])/255               
                    red_color = [state_of_visited,state_of_visited,state_of_visited]
                    base_map[:,i,j] = red_color
                else:
                    base_map[:,i,j] = blue_color
        return base_map
    
    def action_space_sample(self):
        return np.random.randint(0,8)

    def set_test_mode(self,test_mode = False):
        self.test_mode = test_mode
        if(test_mode == True):
            print("测试模式已激活")
        else:
            print("测试模式未激活")
        
    # 指标 #
    def metrics(self):    
        if self.test_mode == False:
            print("测试模式未激活。此调用将不会产生效果")
            return -1
        
        # 覆盖率指标 #        
        num_celdas_cubiertas = np.count_nonzero(self.visited)
        num_celdas_visitables = np.count_nonzero(self.map)
        
        coverage = num_celdas_cubiertas/num_celdas_visitables
        
        # 访问频率指标 #        
        V = []
        # 获取每个访问的等待时间向量 #
        T_mean_M = self.mean_time_M/self.visited_M
        
        for i in range(0,self.map.shape[0]):
            for j in range(0,self.map.shape[1]):            
                if(self.map[i][j] == 1 and T_mean_M[i][j] != np.inf and T_mean_M[i][j] > 0):
                    V.append(T_mean_M[i][j])
        
        
        # 计算平均值 #                    
        mean = np.mean(V)
        
        # 标准差 #
        std = np.std(V)
        
        metricsD = {}
        metricsD['coverage'] = coverage
        metricsD['mean'] = mean
        metricsD['std'] = std
        metricsD['time_matrix'] = T_mean_M
        metricsD['visited_acc_matrix'] = self.visited_M
        metricsD['V'] = V
        
        return metricsD
    
        
        
        
        
        
                    
                
        
        
        
        
        
        
        
            
        