#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import genfromtxt

class Environment:
    
    def __init__(self):
        self.map = genfromtxt('./YpacaraiMap.csv', delimiter=',',dtype = int)
        self.S = np.ones(self.map.shape)
        self.R_abs = np.ones(self.map.shape)*255
        self.visited = np.zeros(self.map.shape)
        
        self.visited_M = np.zeros(shape = self.map.shape)
        self.waiting_M = np.ones(shape = self.map.shape)
        self.mean_time_M = np.zeros(shape = self.map.shape)
        
        self.test_mode = False
        
        posible_x, posible_y = np.nonzero(self.map)        
        init_cell_index = np.random.randint(0,posible_x.size)        
        self.agent_start_pos = (posible_x[init_cell_index],posible_y[init_cell_index])

        self.agent_pos = self.agent_start_pos
        self.agent_pos_ant = self.agent_start_pos
        
        self.visited[self.agent_pos[0]][self.agent_pos[1]] = 255
        self.S[self.agent_pos[0]][self.agent_pos[1]] = 0
        self.R_abs[self.agent_pos[0]][self.agent_pos[1]] -= 50
        
    def reset(self):
        self.S = np.ones(self.map.shape)
        self.R_abs = np.ones(self.map.shape)*255
        self.visited = np.zeros(self.map.shape)
        
        self.visited_M = np.zeros(shape = self.map.shape)
        self.waiting_M = np.ones(shape = self.map.shape)
        self.mean_time_M = np.zeros(shape = self.map.shape)
        
        posible_x, posible_y = np.nonzero(self.map)
        init_cell_index = np.random.randint(0,posible_x.size)
        
        self.agent_start_pos = [posible_x[init_cell_index],posible_y[init_cell_index]]

        self.agent_pos = self.agent_start_pos
        self.agent_pos_ant = self.agent_start_pos
        
        self.visited[self.agent_pos[0]][self.agent_pos[1]] = 255
        self.S[self.agent_pos[0]][self.agent_pos[1]] = 0
        self.R_abs[self.agent_pos[0]][self.agent_pos[1]] -= 50
        
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
            
        ilegal = 0
        if self.map[future_pos[0]][future_pos[1]] == 0:
            ilegal = 1
        else:
            self.agent_pos = future_pos
            
        self.visited[self.agent_pos_ant[0]][self.agent_pos_ant[1]] = 127
        self.visited[self.agent_pos[0]][self.agent_pos[1]] = 255
        

        self.R_abs[self.agent_pos_ant[0]][self.agent_pos_ant[1]] =  np.max([self.R_abs[self.agent_pos_ant[0]][self.agent_pos_ant[1]]-50,0])
        

        
        rho_next = self.R_abs[self.agent_pos[0]][self.agent_pos[1]] * self.S[self.agent_pos[0]][self.agent_pos[1]]
        rho_act = self.R_abs[self.agent_pos_ant[0]][self.agent_pos_ant[1]] * self.S[self.agent_pos_ant[0]][self.agent_pos_ant[1]]
        

        
        reward = rho_next - rho_act

        reward = (1-ilegal)*((1.507/255)*(reward-255)+1) - ilegal*(0.5)
        
        
        for i in range(0,self.map.shape[0]):
            for j in range(0,self.map.shape[1]):
                
                self.S[i][j] = np.min([self.S[i][j]+0.0055, 1])
                
        self.S[self.agent_pos[0]][self.agent_pos[1]] = 0
        
        self.agent_pos_ant = self.agent_pos
        
        
        obs = {}
        obs['visited_map'] = self.visited
        obs['importance_map'] = self.S*self.R_abs
        obs['position'] = self.agent_pos
        
        done = 0
        
        if(self.test_mode == True):
            self.visited_M[self.agent_pos[0]][self.agent_pos[1]] += 1
            T = self.waiting_M[self.agent_pos[0]][self.agent_pos[1]]
            self.waiting_M +=  1
            self.waiting_M[self.agent_pos[0]][self.agent_pos[1]] = 1
            self.mean_time_M[self.agent_pos[0]][self.agent_pos[1]] += T
            
        else:
            pass
        
        return obs, reward, done, ilegal
    
    def render(self):
        
        green_color = np.asarray([0,160,20])/255
        blue_color = np.asarray([0,0,0])/255
        agent_color = np.asarray([255,0,0])/255
        red_color = np.asarray([241,241,241])/255
        
        size_map = (self.map.shape[0],self.map.shape[1],3)
        base_map = np.zeros(size_map)
        
        for i in range(0,self.map.shape[0]):
            for j in range(0,self.map.shape[1]):
                
                if(self.map[i][j] == 0):
                    base_map[i][j] = green_color
                elif(self.agent_pos[0] == i and self.agent_pos[1] == j):
                    base_map[i][j] = agent_color
                elif(self.visited[i][j] != 0):
                    state_of_visited = (255-self.R_abs[i][j]*self.S[i][j])/255               
                    red_color = [state_of_visited,state_of_visited,state_of_visited]
                    base_map[i][j] = red_color
                else:
                    base_map[i][j] = blue_color
        
        return base_map
    
    def action_space_sample(self):
        
        return np.random.randint(0,8)

    def set_test_mode(self,test_mode = False):
        self.test_mode = test_mode

        if(test_mode == True):
            print("Modo test activado")
        else:
            print("Modo test desactivado")

    
    
    def metrics(self):
        
        if self.test_mode == False:
            print("El modo test no se ha activado. Esta llamada no tendrá efecto")
            return -1
        
        num_celdas_cubiertas = np.count_nonzero(self.visited)
        num_celdas_visitables = np.count_nonzero(self.map)
        
        coverage = num_celdas_cubiertas/num_celdas_visitables
        
        V = []
        T_mean_M = self.mean_time_M/self.visited_M
        
        for i in range(0,self.map.shape[0]):
            for j in range(0,self.map.shape[1]):
                
                if(self.map[i][j] == 1 and T_mean_M[i][j] != np.inf and T_mean_M[i][j] > 0):
                    V.append(T_mean_M[i][j])
        
        
        mean = np.mean(V)
        std = np.std(V)
        
        metricsD = {}
        metricsD['coverage'] = coverage
        metricsD['mean'] = mean
        metricsD['std'] = std
        metricsD['time_matrix'] = T_mean_M
        metricsD['visited_acc_matrix'] = self.visited_M
        metricsD['V'] = V
        
        return metricsD
    
        
        
        
        
        
                    
                
        
        
        
        
        
        
        
            
        