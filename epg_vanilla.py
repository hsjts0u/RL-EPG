# Spring 2021, Reinforcement Learning
# Team Implementation Project: Expected Policy Gradient

import gym
import numpy as np
import torch
import torch.nn as nn
import numdifftools as nd
import torch.nn.functional as F
import pickle
from torch.optim import Adam
from copy import deepcopy
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from itertools import count

from model import (Actor, Critic)
from graphing import learning_curve

class EPG(object):
    def __init__(self, n_states, n_actions, action_high, action_low):
        self.n_states = n_states
        self.n_actions = n_actions
        self.total_step = 100_000
        self.max_eps = 100_000
        
        self.LR = 0.001
        self.sigma = 0.5
        self.discount = 0.99
        self.c = 1.0
        
        self.action_high = action_high
        self.action_low = action_low

        self.actor = Actor(self.n_states, self.n_actions)
        self.actor_optim  = Adam(self.actor.parameters(), lr=self.LR)

        self.critic = Critic(self.n_states, self.n_actions)
        self.critic_optim  = Adam(self.critic.parameters(), lr=self.LR)


    def getHessian(self, state, mu):
        Q = lambda mu: self.critic(torch.cat((torch.Tensor(state), \
                        torch.Tensor(mu)), dim=-1)).detach().numpy()
        H = nd.Hessian(Q)(mu.detach().numpy())
        return H[0][0]
    
    # this function returns sigma1/2 in the paper's notation
    def covariance(self, critic, state, mu, printH=False):
        Q = lambda mu: critic(torch.cat((torch.Tensor(state), \
                        torch.Tensor(mu)), dim=-1)).detach().numpy()
        
        # take hessian of Q with respect to mu
        H = nd.Hessian(Q)(mu.detach().numpy())
        if printH:
            print(H)
        eig, v = np.linalg.eig(H)
        return eig, self.sigma * torch.matrix_exp(self.c * \
            torch.Tensor(H))
    
    
    def get_action(self, mu, cov):
        mu = mu.detach()
        
        # next line use sigma1/2 * sigma1/2 as covariance matrix
        action_dis = MultivariateNormal(mu, torch.mm(cov, cov) + 0.0000001 * torch.Tensor(np.eye(self.n_actions)))
        
        # sample an action
        action = action_dis.sample()
        
        # if the sampled action is out of range, clip 
        return torch.clip(action, 
            min=self.action_low, 
            max=self.action_high)

    def save_model(self, name):
        torch.save(
            self.actor.state_dict(),
            f'epg/epg_actor_{name}.pth'
        )
        torch.save(
            self.critic.state_dict(),
            f'epg/epg_critic_{name}.pth'
        )

    def load_model(self, name):
        self.actor.load_state_dict(
            torch.load(f'epg/epg_actor_{name}.pth')
        )

        self.critic.load_state_dict(
            torch.load(f'epg/epg_critic_{name}.pth')
        )
    
    
    def train(self, env):
        
        ewma_reward = 0
        step_count = 0
        episode_reward_list = []
        
        for episode in range(self.max_eps):
            
            if step_count >= self.total_step:
                break
            
            state = env.reset()
            ep_reward = 0
            gamma_accum = 1.0
            
            # get initial mean and sigma1/2
            mu = self.actor(torch.Tensor(state)) * \
                    self.action_high
            #cov = self.covariance(self.critic, state, mu)
            
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
                eig, cov = self.covariance(self.critic, state, mu)
                eigs.append((step_count,eig))
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
                # if step == 999:
                #    target = self.c * reward + self.discount * next_Q
                
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

                # if step_count > 1000 and step > 600:
                #    self.save_model(f'vanilla_{step_count}')
                    
                    # qlist = []
                    # hlist = []
                    # act = np.array([env.action_space.low[0]], dtype=np.float32)
                    # state_tensor = torch.tensor(state, dtype=torch.float32)
                    # amean = 3.0 * self.actor(state_tensor)
                    # print(amean)
                    # self.covariance(self.critic, state, amean, printH=True)
                    # while act < env.action_space.high[0]:
                    #    qval = self.critic(torch.cat((state_tensor,
                    #                                  torch.tensor(act)),
                    #                                 dim=-1))
                    #    hess = self.getHessian(state, torch.tensor(act))
                        # qlist.append((act[0], qval.item()))
                        # hlist.append((act[0], hess))
                    #    act = act + 1e-3
                    # qf = open('qvalue.pickle', 'wb')
                    # pickle.dump(qlist, qf)
                    # qf.close()
                    # hf = open('hessian.pickle', 'wb')
                    # pickle.dump(hlist, hf)
                    # hf.close()
                    # return eigs

                
                if done:
                    # when the episode is finished print and save some results for graphing
                    # restart a new episode and continue training
                    # episode_reward_list.append((step_count, ep_reward))
                    ewma_reward = 0.05 * ep_reward + 0.95 * ewma_reward
                    episode_reward_list.append((step_count, ep_reward))
                    print(f'Steps {step_count}\tEpisode {episode}\tlength: {step+1}\treward: {ep_reward}\t ewma reward: {ewma_reward}')
                    break
                
        return episode_reward_list
    
def save_learning_curve(data, file_name):
    """ data should be list of (step, episode_rewards) """
    f = open(file_name, "a")
    for step, reward in data:
        f.write(f"{step} {reward}\n")
    f.write('0 0\n')
    f.close()
    
if __name__ == '__main__':
    random_seed = 10
    env_list = ['HalfCheetah-v2', 'InvertedPendulum-v2',
                'Reacher-v2', 'Walker2d-v2']
    
    env = gym.make('InvertedPendulum-v2')
    env.seed(random_seed)

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    for _ in range(1):
        model = EPG(observation_dim, action_dim, 
                env.action_space.high[0], env.action_space.low[0])
        # model.load_model('vanilla_1265')
        data = model.train(env)    
        save_learning_curve(data, "epg/epg_vanilla_ip.txt")
