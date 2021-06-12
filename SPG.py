# SPG

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from itertools import count

from model import (Actor, Critic)

USE_CUDA = torch.cuda.is_available()

LR = 0.001
SIGMA = 0.2
BATCH_SIZE = 64
DISCOUNT = 0.99

MAX_EPISODE = 20

loss = nn.MSELoss()

def to_numpy(var):
    """
    turn pytorch tensor to numpy array
    """
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, requires_grad=False, dtype=torch.float32):
    """ turn numpy array to pytorch tensor  """
    return torch.tensor(torch.from_numpy(ndarray),
                        dtype=dtype, requires_grad=requires_grad)


class SPG(object):
    def __init__(self, nb_states, nb_actions):
        self.nb_states = nb_states
        self.nb_actions = nb_actions

        self.actor = Actor(self.nb_states, self.nb_actions)
        self.actor_optim = Adam(self.actor.parameters(), lr=LR)

        self.critic = Critic(self.nb_states, self.nb_actions)
        self.critic_optim = Adam(self.critic.parameters(), lr=LR)

        self.batch_size = BATCH_SIZE
        self.discount = DISCOUNT

        if USE_CUDA:
            self.cuda()
    
    def cuda(self):
        self.actor.cuda()
        self.critic.cuda()
    
    def train(self, num_iter, env, max_episode_length=None):
        for episode in range(MAX_EPISODE):
            state = env.reset()
            episode_score = 0
            s_a = []
            
            for episode_steps in count():
                mean = self.actor(to_tensor(state))
                action = torch.clip(
                            torch.normal(mean, SIGMA),
                            float(env.action_space.low[0]), 
                            float(env.action_space.high[0]))
                next_state, reward, done, _ = env.step(to_numpy(action))
                episode_score += reward 
                
                s_a.append((state, action, reward, mean))
                state = next_state
                
                if len(s_a) > self.batch_size or done:
                    self.update_actor(s_a)
                    self.update_critic(s_a)
                    s_a.clear()
                
                if done:
                    print(f"{episode}: episode_score is {episode_score}")
                    break
                    
    def update_actor(self, s_a):
        gamma_accum = 1
        loss = 0
        for (state, action, reward, mean) in s_a:
            action_dist = torch.distributions.normal.Normal(mean, 0.2)
            sa_cat = torch.cat((torch.Tensor(state), action), dim=-1)
            loss += gamma_accum * self.critic(sa_cat).detach_() * \
                        -action_dist.log_prob(action)
            gamma_accum *= self.discount
        
        loss = loss.mean()
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
    
    def update_critic(self, s_a):
        loss = 0
        for i in range(len(s_a) - 1):
            sa_cat = torch.cat((torch.Tensor(s_a[i][0]), s_a[i][1].detach()), dim=-1)
            sa_n_cat = torch.cat((torch.Tensor(s_a[i+1][0]), s_a[i+1][1].detach()), dim=-1)
            target = s_a[i][2] + self.discount * self.critic(sa_n_cat).detach_()
            loss += F.mse_loss(self.critic(sa_cat), target)
            
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
                
                
if __name__ == "__main__":
    env = gym.make("HalfCheetah-v2")
    model = SPG(env.observation_space.shape[0], env.action_space.shape[0])
    model.train(20, env, 280000)
                
                
                
                
                
            
