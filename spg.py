# SPG Q-actor-critic
 
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torch.optim import Adam
from itertools import count
from torch.distributions.normal import Normal
from copy import deepcopy
 
from model import (Actor, Critic)
from graphing import learning_curve
 
class SPG(object):
    def __init__(self, actor_lr, critic_lr, gamma, nb_states, nb_actions,
                 max_episodes, max_ep_steps, reward_scaling, max_steps):
        
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.max_ep_steps = max_ep_steps
        self.reward_scaling = reward_scaling
        self.max_steps = max_steps
        
        self.actor = Actor(nb_states, nb_actions)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = Critic(nb_states, 0)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        
        
    def train(self, env):
        ewma_rewards = 0.0
        step_count = 0
        graph_data = []
        
        for eps in range(self.max_episodes):
            
            if step_count >= self.max_steps:
                break
            
            eps_score = 0
            state = env.reset()
            
            for step in range(self.max_ep_steps):
                step_count += 1
                action, log_prob = self.choose_action(state, env)
                next_state, reward, done, _ = env.step(action)
                
                eps_score += reward
                reward *= self.reward_scaling
                
                self.learn(state, action, reward, next_state, log_prob, done)
 
                state = next_state
 
                if done:
                    graph_data.append((step_count, eps_score))
                    ewma_rewards = 0.05 * eps_score + 0.95 * ewma_rewards
                    
                    if (eps % 100) == 0:
                        print('Steps {}\tEpisode {}\tlength: {}\tewma_reward: {}'.format(\
                            step_count, eps, step+1, ewma_rewards))
                    break
            
        return graph_data
        
        
    def choose_action(self, state, env):
        mean = self.actor(torch.Tensor(state)) * env.action_space.high[0]
        action_dist = Normal(mean, 0.2)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        action = torch.clip(action, \
            float(env.action_space.low[0]), float(env.action_space.high[0]))
        
        return action.detach().numpy(), log_prob
           
           
    def learn(self, state, action, reward, next_state, log_prob, done):
        s = torch.Tensor(state)
        ns = torch.Tensor(next_state)
 
        v = self.critic(s)
        nv = self.critic(ns)
        
        actor_loss = (reward + self.gamma * nv.detach() - v.detach())* -log_prob
        critic_loss = F.mse_loss(reward + self.gamma * nv.detach() * (1-int(done)), v)
        
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        
        actor_loss.mean().backward()
        critic_loss.backward()
        
        self.actor_optim.step()
        self.critic_optim.step()
        

def save_learning_curve(data, file_name):
    """ data should be list of (step, episode_rewards) """
    f = open(file_name, "a")
    f.write('[ ')
    for step, reward in data:
        f.write(f"({step}, {reward}),")
    f.write(']\n')
    f.close()

if __name__ == "__main__":
    envs = (("InvertedPendulum-v2", 0.1), 
            ("HalfCheetah-v2", 1),
            ("Reacher2d-v2", 1),
            ("Walker2d-v2", 1))
    env = gym.make(envs[0][0])
    env.seed(20)
    
    
    f = open("spg_inverted_pendulum.txt", "w")
    f.close()
    curves = []
    for i in range(40):
        model = SPG(0.001, 0.001, 0.9, env.observation_space.shape[0],\
            env.action_space.shape[0], 100_000, 1000, envs[0][1], 100_000)
        data = model.train(env)
        #curves.append(data)
        save_learning_curve(data, "spg_inverted_pendulum.txt", "ddpg_ip")
    
    
    #learning_curve(curves, "spg-InvertedPendulum", "./spg")
    
    
    
    
    
    
    
    
