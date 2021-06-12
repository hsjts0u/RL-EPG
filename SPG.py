# SPG

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from itertools import count

from model import (Actor, Critic)

LR = 0.001
SIGMA = 0.2
BATCH_SIZE = 64
DISCOUNT = 0.99

MAX_EPISODE = 1000


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

    def train(self, env):
        for episode in range(MAX_EPISODE):
            state = env.reset()
            episode_score = 0
            s_a = []
            
            for episode_steps in count():
                mean = self.actor(torch.Tensor(state))
                action = torch.clip(
                            torch.normal(mean, SIGMA),
                            float(env.action_space.low[0]), 
                            float(env.action_space.high[0]))
                next_state, reward, done, _ = env.step(action.data.numpy())
                episode_score += reward 
                
                s_a.append((state, action, reward, mean, done))
                state = next_state
                
                if len(s_a) > self.batch_size or done:
                    self.update_actor(s_a)
                    self.update_critic(s_a)
                    s_a.clear()

                if done:
                    print(f"{episode}: episode_score is {episode_score}, episode_length is {episode_steps}")
                    break
                    
    def update_actor(self, s_a):
        gamma_accum = 1
        loss = 0
        for (state, action, reward, mean, done) in s_a:
            action_dist = torch.distributions.normal.Normal(mean, 0.2)
            sa_cat = torch.cat((torch.Tensor(state), action), dim=-1)
            if not done:
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
            if not s_a[i+1][4]:
                target = s_a[i][2] + self.discount * self.critic(sa_n_cat).detach_()
            else:
                target = s_a[i][2]
            loss += F.mse_loss(self.critic(sa_cat), torch.Tensor([target]))
            
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
                
                
if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")
    env.seed(20)
    torch.manual_seed(20)
    model = SPG(env.observation_space.shape[0], env.action_space.shape[0])
    model.train(env)
                
                
                
                
                
            
