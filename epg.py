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

def Gauss_integral(Q, state, mu):
    I = mu * Q
    return I

def Get_Covariance(Q, mu):
    H = hessian(Q, mu)
    return sigma_0 * torch.exp(c * H)

class EPG(object):
    def __init__(self, n_states, n_actions, sigma=0.5):
        self.lr = lr
        self.n_states = n_states
        self.n_actions = n_actions
        self.std = sigma

        self.actor = Actor(self.n_states, self.n_actions)
        self.critic = Critic(self.n_states, self.n_actions)

        self.actor_optim  = Adam(self.actor.parameters(), lr=LR)
        self.critic_optim  = Adam(self.critic.parameters(), lr=LR)
        
        if USE_CUDA:
            self.cuda()

    def select_action(self, mu):
        action = torch.normal(mean=mu, std=self.std, size=1)
        return torch.clamp(action, min=-1, max=1)

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

def train(env, ewma_threshold, lr=0.01):

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]
    
    model = EPG(observation_dim, action_dim)
    
    ewma_threshold = 0
    gamma = 0.999

    for i_episode in count(1):
        state = env.reset()
        ep_reward = 0
        t = 0
        gamma_t = 1
        for t in range(10000):
            mu = model.actor(state)
            action = model.select_action(mu)

            Q = model.critic(torch.cat((state, action), dim=1))
            g_t = gamma_t * Gauss_integral(Q, s, mu)

            model.actor_optim.zero_grad()
            g_t.backward()
            model.actor_optim.step()

            model.std = Get_Covariance(Q, mu)
            
            new_state, reward, done, _ = env.step(action)

            ep_reward += reward
            gamma_t *= gamma

        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print(f'Episode {i_episode}\tlength: {t}\treward: {ep_reward}\t ewma reward: {ewma_reward}')

        if ewma_reward > ewma_threshold:
            model.save_model()
            print(f"Solved! Running reward is now {ewma_reward} and the last episode runs to {t} time steps!")
            break


def test(env, model_name, n_episodes=10):

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]
    
    model = EPG(observation_dim, action_dim)
    model.load_weights()
    
    render = True

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            mu = model.actor(state)
            action = model.select_action(mu)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        print(f'Episode {i_episode}\tReward: {running_reward}')

    env.close()


if __name__ == '__main__':
    random_seed = 20
    # env_list = ['HalfCheetah-v2', 'InvertedPendulum-v2',
                # 'Reacher2d-v2', 'Walker2d-v2']
    env_list = ['CartPole-v0']
    for e in env_list:
        env = gym.make(e)
        env.seed(random_seed)
        torch.manual_seed(random_seed)

        train(env, ewma_threshold, lr)
        test()
