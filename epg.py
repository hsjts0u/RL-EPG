# Spring 2021, Reinforcement Learning
# Team Implementation Project: Expected Policy Gradient

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd.functional import hessian
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from itertools import count

from model import (Actor, Critic)
from graphing import learning_curve

USE_CUDA = torch.cuda.is_available()

sigma_0 = 0.5
c = 1.0
LR = 0.001
DISCOUNT = 0.99
MAX_EPISODE = 25000
TOTAL_STEP = 1000000

loss = nn.MSELoss()

def Get_Covariance(critic, state, a):
    Q_function = lambda action: critic(torch.cat((to_tensor(state), action)))
    H = hessian(Q_function, a)
    return sigma_0 * torch.exp(c * H).squeeze(0)

class EPG(object):
    def __init__(self, n_states, n_actions, action_high, action_low):
        self.n_states = n_states
        self.n_actions = n_actions
        self.std = torch.tensor([sigma_0])
        self.discount = DISCOUNT
        self.total_step = TOTAL_STEP
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

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(), 
            f'./{output}/epg_actor.pth'
        )
        torch.save(
            self.critic.state_dict(), 
            f'./{output}/epg_critic.pth'
        )

    def load_weights(self, output):
        if output is None:
            return

        self.actor.load_state_dict(
            torch.load(f'./{output}/epg_actor.pth')
        )
        self.critic.load_state_dict(
            torch.load(f'./{output}/epg_actor.pth')
        )

    def seed(self, s):
        torch.manual_seed(s)
        if self.USE_CUDA:
            torch.cuda.manual_seed(s)

    def select_action(self, mu):
        if mu.ndimension() == 1:
            action_dis = Normal(mu, self.std)
        else:
            action_dis = MultivariateNormal(mu, self.std)
        action = action_dis.sample()

        return torch.clamp(action, 
            min=self.action_low, 
            max=self.action_high)

    def train(self, env):
        
        ewma_reward = 0
        step_count = 0
        episode_reward_list = []

        for episode in range(MAX_EPISODE):

            if step_count >= self.total_step:
                break

            state = env.reset()
            ep_reward = 0
            gamma_accum = 1
            step = 0
            for step in range(10000):
                step_count += 1

                mu = self.actor(torch.Tensor(state)) * self.action_high
                action = self.select_action(mu)
                Q = self.critic(torch.cat((torch.Tensor(state), mu), dim=-1))
                
                g_t = gamma_accum * Q

                self.actor_optim.zero_grad()
                g_t.backward()
                self.actor_optim.step()

                self.std = Get_Covariance(self.critic, state, action)
                new_state, reward, done, _ = env.step(action.detach().numpy())

                next_q = self.critic(
                    torch.cat((
                        torch.Tensor(new_state), 
                        self.actor(torch.Tensor(new_state)).detach()
                    ), dim=-1))
                target_q = reward + self.discount * done * next_q

                mu = self.actor(torch.Tensor(state)) * self.action_high
                action = self.select_action(mu)
                Q = self.critic(torch.cat((torch.Tensor(state), action), dim=-1))
                
                critic_loss = loss(Q, target_q)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                ep_reward += reward
                gamma_accum *= self.discount
                state = new_state

                if done:
                    episode_reward_list.append((step_count, ep_reward))
                    ewma_reward = 0.05 * ep_reward + 0.95 * ewma_reward
                    # self.save_model()
                    if episode % 100 == 0:
                        print(f'Episode {episode}\tlength: {step+1}\treward: {ep_reward}\t ewma reward: {ewma_reward}')
                    break

        return episode_reward_list

    def test(self, env, n_episodes=10):
        
        self.load_weights()
        
        render = True

        for episode in range(1, n_episodes+1):
            state = env.reset()
            running_reward = 0
            for t in range(10000):
                mu = self.actor(torch.Tensor(state))
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
    
    env = gym.make('MountainCarContinuous-v0')
    env.seed(random_seed)

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]

    torch.manual_seed(random_seed)
    model = EPG(observation_dim, action_dim, 
                env.action_space.high[0], env.action_space.low[0])
    
    episode_reward_list = model.train(env)
    learning_curve([episode_reward_list], "epg")
    # model.test(env)
