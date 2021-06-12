# SPG

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from itertools import count

from .model import (Actor, Critic)

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
    return torch.tensor(torch.from_numpy(ndarray=ndarray),
                        dtype=dtype, requires_grad=requires_grad)

class Memory(object):
    def __init__(self, memory_size=10000):
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size

    def __len__(self):
        return len(self.memory)

    def append(self, item):
        self.memory.append(item)

    def sample_batch(self, batch_size):
        idx = np.random.permutation(len(self.memory))[:batch_size]
        return [self.memory[i] for i in idx]


class SPG(object):
    def __init__(self, nb_states, nb_actions):
        self.nb_states = nb_states
        self.nb_actions = nb_actions

        self.actor = Actor(self.nb_states, self.nb_actions)
        self.actor_optim = Adam(self.actor.parameters(), lr=LR)

        self.critic = Critic(self.nb_states, self.nb_actions)
        self.critic_optim = Adam(self.critic.parameters(), lr=LR)
        
        self.memory = Memory(10000)

        self.batch_size = BATCH_SIZE
        self.discount = DISCOUNT

        if USE_CUDA:
            self.cuda()
    
    def cuda(self):
        self.actor.cuda()
        self.critic.cuda()
    
    def train(self, num_iter, env, output, max_episode_length=None):
        for episode in range(MAX_EPISODE):
            state = env.reset()
            episode_score = 0
            s_a = []
            
            for episode_steps in count():
                action = torch.clip(
                            torch.normal(
                                self.actor(to_tensor(state)), SIGMA
                            ),
                            float(env.action_space.low[0]), 
                            float(env.action_space.high[0]))
                next_state, reward, done, _ = env.step(to_numpy(action))
                episode_score += reward 
                
                s_a.append((state, action, reward))
                state = next_state
                
                if len(s_a) > self.batch_size or done:
                    update_actor(s_a)
                    update_critic(s_a)
                    s_a.clear()
                    
    def update_actor(s_a):
        gamma_accum = 1
        loss = 0
        for (state, action, reward) in s_a:
            action_dist = torch.distributions.normal.Normal(to_tensor(action), 0.2)
            sa_cat = torch.cat((to_tensor(state), to_tensor(action)), dim=1)
            loss += gamma_accum * self.critic(sa_cat) * \
                        -action_dist.log_prob(to_tensor(action))
            gamma_accum *= self.discount
        
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
    
    def update_critic(s_a):
        loss = 0
        for i in (len(s_a) - 1):
            sa_cat = torch.cat((to_tensor(s_a[i].state), to_tensor(s_a[i].action)), dim=1)
            sa_n_cat = torch.cat((to_tensor(s_a[i+1].state), to_tensor(s_a[i+1].action)), dim=1)
            target = r + self.discount * self.critic(sa_n_cat)
            loss += F.mse_loss(self.critic(sa_cat), target)
                
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
                
                
                
                
                
                
                
                
                
            
