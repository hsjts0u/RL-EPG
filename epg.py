import torch
import torch.nn as nn
import gym
import numpy as np
import numdifftools as nd
from copy import deepcopy
from torch.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from model import Actor, Critic
from graphing import learning_curve

LR = 0.001
DISCOUNT = 0.99
SIGMA = 0.2
C = 1.0
EIGEN_CLIP = 1

def to_tensor(ndarray, requires_grad=False, numpy_dtype=np.float32):
    """ turn numpy array to pytorch tensor  """
    t = torch.from_numpy(ndarray.astype(numpy_dtype))
    if requires_grad:
        t.requires_grad_()
    return t

class epg:
    def __init__(self, state_dim, action_dim, action_lim):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.actor_optim = Adam(self.actor.parameters(), lr=LR)
        self.critic_optim = Adam(self.critic.parameters(), lr=LR)
    
    def action_mean(self, state):
        return self.actor(to_tensor(state)) * self.action_lim
    
    def select_action(self, mean, covaraince):
        act = MultivariateNormal(mean.double(), covaraince).sample().float()
        act = torch.clamp(act, -self.action_lim, self.action_lim)
        return act
    
    def expm(self, m):
        """ this function can only handle symmetric matrix in numpy """
        w, v = np.linalg.eig(m)
        w = 2.0 * np.clip(w, None, EIGEN_CLIP)
        expw = np.exp(w)
        # print(expw)
        return v @ np.diag(expw) @ v.T
    
    def get_covariance(self, state, mu):
        """ mu should be a 1d torch tensor """
        def Qsu(act):
            return self.critic(torch.cat((to_tensor(state), to_tensor(act)), dim=-1)).detach().numpy()
        
        cH = C * nd.Hessian(Qsu)(mu.detach().numpy()).astype(np.float64)
        cov = SIGMA * self.expm(cH)
        return torch.from_numpy(cov) + 0.01 * torch.eye(self.action_dim, dtype=torch.float64)
        
    def train(self, env, step_limit):
        
        obs = env.reset()
        
        mu = self.action_mean(obs)
        action = self.select_action(mu, self.get_covariance(obs, mu))
        
        gamma_acc = 1.0
        episode_reward = 0.0
        episode = 1
        episode_step = 0
        
        learn_data = []
        
        loss = nn.MSELoss()
        
        for step in range(step_limit):
            episode_step += 1
            actor_loss = -gamma_acc * self.critic(torch.cat((to_tensor(obs), mu), dim=-1))
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            obs2, reward, done, _ = env.step(action.detach().numpy())
            env.render()
            mu2 = self.action_mean(obs2)
            action2 = self.select_action(mu2.detach(), self.get_covariance(obs2, mu2))
            target_q = reward + DISCOUNT * (1-int(done)) * \
                self.critic(torch.cat((to_tensor(obs2), action2.detach()), dim=-1))
            q = self.critic(torch.cat((to_tensor(obs), action.detach()), dim=-1))
            critic_loss = loss(target_q.detach(), q)
            
            # print(critic_loss)
            
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            
            obs = deepcopy(obs2)
            mu = mu2
            action = action2
            
            episode_reward += reward
            gamma_acc *= DISCOUNT
            
            if done:
                
                print(f'step {step} episode {episode} reward {episode_reward} in {episode_step} steps')
                
                learn_data.append((step, episode_reward))
                gamma_acc = 1.0
                episode += 1
                episode_reward = 0
                episode_step = 0
                
                
                obs = env.reset()
                mu = self.action_mean(obs)
                action = self.select_action(mu, self.get_covariance(obs, mu))
        
        return learn_data
                
if __name__ == '__main__':
    
    env = gym.make('BipedalWalker-v3')
    agent = epg(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
    curve = []
    data = agent.train(env, 300_000)
    curve.append(data)
    learning_curve(curve, "epg4_bipedalwalker.png")
