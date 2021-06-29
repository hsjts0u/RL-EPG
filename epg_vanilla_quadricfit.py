# Spring 2021, Reinforcement Learning
# Team Implementation Project: Expected Policy Gradient

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
from torch.distributions.uniform import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal

from model import (Actor, Critic)
from graphing import learning_curve

class EPG(object):
    def __init__(self, total_step, n_states, n_actions, action_high, action_low):
        self.n_states = n_states
        self.n_actions = n_actions
        self.total_step = total_step
        self.max_eps = 100_000
        
        self.LR = 0.001
        self.sigma = 0.5
        self.discount = 0.99
        self.c = 1.0
        
        self.action_high = action_high
        self.action_low = action_low
        self.n_actions = n_actions

        self.actor = Actor(self.n_states, self.n_actions)
        self.actor_optim  = Adam(self.actor.parameters(), lr=self.LR)

        self.critic = Critic(self.n_states, self.n_actions)
        self.critic_optim  = Adam(self.critic.parameters(), lr=self.LR)

    
    def expm(self, m):
        """ this function can only handle symmetric matrix in numpy """
        with torch.no_grad():
            L, V = torch.symeig(m, eigenvectors=True)
            L = torch.clamp_max(L, 1.0)
            expw = torch.exp(2.0 * L)
            return torch.matmul(V, torch.matmul(expw.diag_embed(),
                                                V.transpose(-2, -1)))
        
    def get_covariance(self, state, mu):
        """ 
        state : 1d numpy array
        mu : 1d torch tensor
        return action_dim x action_dim torch tensor
        """
    
        state_tensor = torch.tensor(state, dtype=torch.float32)
        num_sample = 100
        sample_range = 0.2
        acts = Uniform(mu-sample_range, mu+sample_range).sample((num_sample,))
        qs = self.critic(torch.cat((state_tensor.repeat(num_sample, 1), acts), dim=-1))

        sample_Q = qs.reshape(num_sample, 1).detach()
        sample_action = torch.transpose(acts, 0, 1)

        
        poly_vars = [torch.ones(num_sample)]
        for i in range(self.n_actions):
            poly_vars.append(sample_action[i])
        for i in range(self.n_actions):
            for j in range(i, self.n_actions):
                poly_vars.append(sample_action[i]*sample_action[j])

        A = torch.stack(poly_vars, dim=1)
        X, _ = torch.lstsq(sample_Q, A)
        idx = 1+self.n_actions
        h = torch.zeros(self.n_actions, self.n_actions)
        for i in range(self.n_actions):
            for j in range(i, self.n_actions):
                h[i][j] = X[idx].item()
                idx += 1
        H = h + torch.transpose(h.clone(), -2, -1)
        cov = self.sigma * self.sigma * self.expm(self.c * H.double())
        return cov.float() + 0.0001 * torch.eye(self.n_actions)

    def test(self, env, step_lim=1_000_000):
        state = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        while not done and step < step_lim:
            step += 1
            act = self.actor(torch.tensor(state, dtype=torch.float32)).detach().numpy() * self.action_high
            new_state, reward, done, _ = env.step(act)
            total_reward += reward
            state = deepcopy(new_state)
        return total_reward
    
    def get_action(self, mu, cov):
        mu = mu.detach()
        
        # next line use sigma1/2 * sigma1/2 as covariance matrix
        action_dis = MultivariateNormal(mu, cov)
        
        # sample an action
        action = action_dis.sample()
        
        # if the sampled action is out of range, clip 
        return torch.clip(action, 
            min=self.action_low, 
            max=self.action_high)
    
    
    def train(self, env, test_env):
        
        ewma_reward = 0
        step_count = 0
        episode_reward_list = []
        test_data = []
        
        for episode in range(self.max_eps):
            
            if step_count >= self.total_step:
                break
            
            state = env.reset()
            ep_reward = 0
            gamma_accum = 1.0
            
            # get initial mean 
            mu = self.actor(torch.Tensor(state)) * \
                    self.action_high
            
            for step in range(10000):
                step_count += 1
                
                # get Q of state and mu to update policy
                gt_Q = self.critic(torch.cat((torch.Tensor(state), mu),
                                        dim=-1))
                
                # g_t used for policy network update 
                g_t = gamma_accum * (-gt_Q)
                
                # update policy
                self.actor_optim.zero_grad()
                g_t.backward()
                self.actor_optim.step()
                
                # sample action using mu, cov
                cov = self.get_covariance(state, mu)
                action = self.get_action(mu, cov)
                
                # calculate Q using state, sampled action
                Q = self.critic(torch.cat((torch.Tensor(state), action.detach()),
                                        dim=-1))
                
                # do action and get next_state
                next_state, reward, done, _ = env.step(action.detach().numpy())
                
                # mu and Q value using next state, next mu
                next_mu = self.actor(torch.Tensor(next_state)) * \
                                        self.action_high
                next_Q = self.critic(torch.cat((torch.Tensor(next_state), next_mu.detach()),
                                        dim=-1))
                
                # calculate the target and mse_loss
                target = self.c * reward + self.discount * (1 - int(done)) * next_Q
                critic_loss = F.mse_loss(Q, target.detach())
                
                # update critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                
                ep_reward += reward
                gamma_accum *= self.discount
                
                # update state and mu
                state = next_state
                mu = next_mu

                if step_count % 1000 == 0:
                    rw = self.test(test_env)
                    test_data.append((step_count, rw))
                
                if done:
                    # when the episode is finished print and save some results for graphing
                    # restart a new episode and continue training
                    ewma_reward = 0.05 * ep_reward + 0.95 * ewma_reward
                    episode_reward_list.append((step_count, ep_reward))
                    print(f'Steps {step_count}\tEpisode {episode}\tlength: {step+1}\treward: {ep_reward}\t ewma reward: {ewma_reward}')
                    break
                
        return episode_reward_list, test_data


if __name__ == '__main__':
    random_seed = 10
    env_list = ['HalfCheetah-v2', 'InvertedPendulum-v2',
                'Reacher-v2', 'Walker2d-v2']
    
    env = gym.make('HalfCheetah-v2')
    env.seed(random_seed)
    test_env = gym.make('HalfCheetah-v2')
    test_env.seed(random_seed)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    curves = []
    tcv = []
    for _ in range(10):
        model = EPG(280_000,
                    observation_dim,
                    action_dim, 
                    env.action_space.high[0],
                    env.action_space.low[0])
        rwl, trwl = model.train(env, test_env)    
        curves.append(rwl)
        tcv.append(trwl)
        for st, rw in trwl:
            print(f'step: {st} reward: {rw}')
        save_learning_curve(rwl, 'epg/epg_vanilla_quadricfit_hc.txt')
        save_learning_curve(trwl, 'epg/epg_vanilla_quaricfit_hc_test.txt')
