# Spring 2021, Reinforcement Learning
# Team Implementation Project: Expected Policy Gradient

import gym
import numpy as np
from itertools import count
from hessian import hessian

import torch
import torch.nn as nn
import torch.optim as optim

from model import (Actor, Critic)

USE_CUDA = torch.cuda.is_available()

sigma_0 = 0.5
c = 1.0
LR = 0.001
DISCOUNT = 0.99
MAX_EPISODE = 20

def to_numpy(var):
    """
    turn pytorch tensor to numpy array
    """
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, requires_grad=False, dtype=torch.float32):
    """
    turn numpy array to pytorch tensor
    """
    return torch.tensor(torch.from_numpy(ndarray),
                        dtype=dtype, requires_grad=requires_grad)

def Gauss_integral(Q, state, mu):
    I = mu * Q
    return I

def Get_Covariance(Q, a):
    H = hessian(Q, a)
    return sigma_0 * torch.exp(c * H)

class EPG(object):
    def __init__(self, n_states, n_actions):
        self.lr = lr
        self.n_states = n_states
        self.n_actions = n_actions
        self.std = sigma_0

        self.actor = Actor(self.n_states, self.n_actions)
        self.critic = Critic(self.n_states, self.n_actions)

        self.actor_optim  = Adam(self.actor.parameters(), lr=LR)
        self.critic_optim  = Adam(self.critic.parameters(), lr=LR)
        
        if USE_CUDA:
            self.cuda()

    def select_action(self, mu):
        action = torch.normal(mean=mu, std=self.std, size=1)
        return torch.clamp(action, 
            min=float(env.action_space.low[0]), 
            max=float(env.action_space.high[0]))

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def cuda(self):
        self.actor.cuda()
        self.critic.cuda()

    def save_model(self):
        torch.save(self.actor.state_dict(), './epg/actor.pth')
        torch.save(self.critic.state_dict(), './epg/critic.pth')

    def load_weights(self):
        self.actor.load_state_dict(torch.load('./epg/actor.pth'))
        self.critic.load_state_dict(torch.load('./epg/actor.pth'))

    def seed(self, s):
        torch.manual_seed(s)
        if self.USE_CUDA:
            torch.cuda.manual_seed(s)

    def train(self, env, ewma_threshold):
        
        ewma_reward = 0
        DISCOUNT = 0.999

        for episode in range(MAX_EPISODE):
            state = env.reset()
            episode_reward = 0
            gamma_accum = 1
            for step in range(10000):
                # state = torch.from_numpy(state).float().unsqueeze(0)
                # mu = model.actor(state)
                mu = self.actor(to_tensor(state))
                action = self.select_action(mu.item())

                Q = self.critic(torch.cat((state, action), dim=1))
                g_t = gamma_accum * Gauss_integral(Q, s, mu)

                self.actor_optim.zero_grad()
                g_t.backward()
                self.actor_optim.step()

                self.std = Get_Covariance(Q, a)
                
                new_state, reward, done, _ = env.step(action.item())

                episode_reward += reward
                gamma_accum *= DISCOUNT
                state = new_state

            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            print(f'Episode {episode}\tlength: {step}\treward: {episode_reward}\t ewma reward: {ewma_reward}')

            if ewma_reward > ewma_threshold:
                self.save_model()
                print(f"Solved! Running reward is now {ewma_reward} and the last episode runs to {t} time steps!")
                break


    def test(self, env, n_episodes=10):
        
        self.load_weights()
        
        render = True

        for episode in range(1, n_episodes+1):
            state = env.reset()
            running_reward = 0
            for t in range(10000):
                mu = self.actor(to_tensor(state))
                action = self.select_action(mu.item())
                state, reward, done, _ = env.step(action)
                running_reward += reward
                if render:
                    env.render()
                if done:
                    break
            print(f'Episode {episode}\tReward: {running_reward}')

        env.close()


if __name__ == '__main__':
    random_seed = 20
    env_list = ['HalfCheetah-v2', 'InvertedPendulum-v2',
                'Reacher2d-v2', 'Walker2d-v2']
    # for e in env_list:
    env = gym.make(env_list[0])
    env.seed(random_seed)

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]

    torch.manual_seed(random_seed)
    model = EPG(observation_dim, action_dim)

    model.train(env, ewma_threshold, lr)
    model.test(env)
