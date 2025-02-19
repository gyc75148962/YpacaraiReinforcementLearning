#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from Qnet import DeepQNetwork
from collections import deque
from replay_memory import ReplayBuffer
from torch.autograd import Variable

class DDQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, 
                 mem_size, batch_size, eps_min, eps_dec, replace):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.update_target_count = replace
        self.action_space = [i for i in range(n_actions)]
        self.replay_memory = deque(maxlen = mem_size)
        
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        
        self.q_eval = DeepQNetwork(self.lr)        
        self.q_next = DeepQNetwork(self.lr)
        
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)        
        

    def sample_memory(self):            
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        with torch.no_grad():
            states = torch.tensor(state).to(self.q_eval.device)
            rewards = torch.tensor(reward).to(self.q_eval.device)
            dones = torch.tensor(done).to(self.q_eval.device)
            actions = torch.tensor(action).to(self.q_eval.device)
            next_state = torch.tensor(new_state).to(self.q_eval.device)

            return states, actions, rewards, next_state, dones
    
    def choose_action_epsilon_greedy(self, observation):
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                observation_array = np.array([observation])
                state = torch.tensor(observation_array,dtype=torch.float).to(self.q_eval.device)
                actions = self.q_eval.forward(state)
                action = torch.argmax(actions).item()
            return action
        else:
            action = np.random.choice(self.action_space)
            return action
    
    def replace_target_network(self,epoch):        
        if epoch % self.update_target_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            
    def learn(self):        
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.q_next.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(next_states)
        q_eval = self.q_eval.forward(next_states)

        max_actions = torch.argmax(q_eval, dim=1).detach()

        q_target = rewards + self.gamma*q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        
    def decrement_epsilon(self):        
        if self.epsilon >=  self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min


        