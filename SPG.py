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
 
torch.autograd.set_detect_anomaly(True)
 
class SPG(object):
    def __init__(self, actor_lr, critic_lr, gamma, nb_states, nb_actions,
                 max_episodes, max_ep_steps, reward_scaling):
        
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.max_ep_steps = max_ep_steps
        self.reward_scaling = reward_scaling
        
        self.actor = Actor(nb_states, nb_actions)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = Critic(nb_states, nb_actions)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        
        
    def train(self, env):
        ewma_rewards = 0.0
        step_count = 0
        
        for eps in range(self.max_episodes):
            eps_score = 0
            state = env.reset()
            
            for step in range(self.max_ep_steps):
                step_count += 1
                
                action, log_prob = self.choose_action(state, env)
                next_state, reward, done, _ = env.step(action)
                next_action, next_log_prob = self.choose_action(next_state, env)
                
                eps_score += reward
                reward *= self.reward_scaling
                
                self.learn(state, action, reward, next_state, \
                    next_action, log_prob, done)
 
                state = next_state
 
                if done:
                    ewma_rewards = 0.05 * eps_score + 0.95 * ewma_rewards
                    if (eps % 100) == 0:
                        print('Steps {}\tEpisode {}\tlength: {}\tewma_reward: {}'.format(\
                            step_count, eps, step+1, ewma_rewards))
                    break
 
        
    def choose_action(self, state, env):
        mean = self.actor(torch.Tensor(state)) * env.action_space.high[0]
        action_dist = Normal(mean, 0.2)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        action = torch.clip(action, \
            float(env.action_space.low[0]), float(env.action_space.high[0]))
        
        return action.detach().numpy(), log_prob
           
           
    def learn(self, state, action, reward, next_state, next_action, log_prob, done):
        qa = torch.cat((torch.Tensor(state), \
            torch.Tensor(action)), dim=-1)
        nqa = torch.cat((torch.Tensor(next_state), \
            torch.Tensor(next_action)), dim=-1)
 
        q = self.critic(qa)
        next_q = self.critic(nqa)
        
        actor_loss = self.gamma * q.detach() * -log_prob
        critic_loss = F.mse_loss(reward + self.gamma * next_q.detach() * (1-int(done)), q)
        
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        
        actor_loss.mean().backward()
        critic_loss.backward()
        
        self.actor_optim.step()
        self.critic_optim.step()
        
        
if __name__ == "__main__":
    envs = (("InvertedPendulum-v2", 0.1), 
            ("HalfCheetah-v2", 1),
            ("Reacher2d-v2", 1),
            ("Walker2d-v2", 1))
    env = gym.make(envs[0][0])
    env.seed(20)
    #print(float(env.action_space.low[0]), float(env.action_space.high[0]))
    model = SPG(0.001, 0.001, 0.9, env.observation_space.shape[0],\
        env.action_space.shape[0], 100000, 1000, envs[0][1])
    model.train(env)
    
    
    
    
    
    
    
    
    
