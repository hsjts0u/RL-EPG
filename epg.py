# Spring 2021, Reinforcement Learning
# Team Implementation Project: Expected Policy Gradient

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
from itertools import count
from hessian import hessian

from model import (Actor, Critic)

USE_CUDA = torch.cuda.is_available()

sigma_0 = 0.5
c = 1.0
LR = 0.001
DISCOUNT = 0.99
MAX_EPISODE = 20

loss = nn.MSELoss()

def hard_update(target, source):
    """
    copy paramerters' value from source to target
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    """
    Update target network with blended weights from target and source.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target.param.data * (1.0 - tau) + param.data * tau
        )

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

def Gauss_integral(Q, mu):
    I = mu * Q
    return I

def Get_Covariance(Q, a):
    H = hessian(Q, a)
    return sigma_0 * torch.exp(c * H)

class EPG(object):
    def __init__(self, n_states, n_actions, action_high, action_low):
        self.n_states = n_states
        self.n_actions = n_actions
        self.std = sigma_0
        self.discount = DISCOUNT
        self.action_high = action_high
        self.action_low = action_low

        self.actor = Actor(self.n_states, self.n_actions)
        self.actor_optim  = Adam(self.actor.parameters(), lr=LR)

        self.critic = Critic(self.n_states, self.n_actions)
        self.critic_optim  = Adam(self.critic.parameters(), lr=LR)

        self.is_training = True
        self.s_t = None
        self.a_t = None
        
        if USE_CUDA:
            self.cuda()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

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

    def select_action(self, mu):
        action_dis = Normal(mu, self.std)
        action = action_dis.sample()

        return torch.clamp(action, 
            min=self.action_low, 
            max=self.action_high)

    def train(self, env, ewma_threshold):
        
        ewma_reward = 0

        for episode in range(MAX_EPISODE):
            state = env.reset()
            ep_reward = 0
            gamma_accum = 1
            for step in range(10000):
                mu = self.actor(to_tensor(state)) * self.action_high
                action = self.select_action(mu)
                Q = self.critic(torch.cat((to_tensor(state), action), dim=-1))
                
                g_t = gamma_accum * Gauss_integral(Q, mu.item())

                self.actor_optim.zero_grad()
                g_t.backward()
                self.actor_optim.step()

                self.std = Get_Covariance(Q, action)
                
                new_state, reward, done, _ = env.step(action.item())

                next_q = self.critic(
                    torch.cat((
                        to_tensor(new_state), 
                        self.actor(to_tensor(new_state)).detach()
                    ), dim=1))
                target_q = reward + self.discount * done * next_q
                critic_loss = loss(Q, target_q)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                ep_reward += reward
                gamma_accum *= self.discount
                state = new_state

            ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
            print(f'Episode {episode}\tlength: {step}\treward: {ep_reward}\t ewma reward: {ewma_reward}')

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
    env = gym.make('MountainCarContinuous-v0')
    env.seed(random_seed)

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]

    torch.manual_seed(random_seed)
    model = EPG(observation_dim, action_dim, 
                env.action_space.high[0], env.action_space.low[0])
    ewma_threshold = 200
    model.train(env, ewma_threshold)
    model.test(env)
