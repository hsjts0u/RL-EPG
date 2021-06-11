# Spring 2021, Reinforcement Learning
# Team Implementation Project: Expected Policy Gradient

import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        """
        WRITE YOUR CODE HERE
        """

    def forward(self, state):
	"""
        WRITE YOUR CODE HERE
        """

    def select_action(self, state):
        """
        WRITE YOUR CODE HERE
        """


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
	"""
        WRITE YOUR CODE HERE
        """

    def forward(self, state):
        """
       WRITE YOUR CODE HERE
        """


def train(model_name, ewma_threshold, lr=0.01):

    mA = Actor()
    mC = Critic()
    optimA = optim.Adam(mA.parameters(), lr=lr)
    optimC = optim.Adam(mC.parameters(), lr=lr)

    ewma_threshold = 0
    gamma = 0.999

    for i_episode in count(1):
        state = env.reset()
        ep_reward = 0
        t = 0

    for t in range(1, 10000):
	"""
        WRITE YOUR CODE HERE
        """

        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print(f'Episode {i_episode}\tlength: {t}\treward: {ep_reward}\t ewma reward: {ewma_reward}')

        if ewma_reward > ewma_threshold:
            torch.save(mA.state_dict(), f'./preTrained/{model_name}_actor.pth')
            torch.save(mC.state_dict(), f'./preTrained/{model_name}_critic.pth')
            print(f"Solved! Running reward is now {ewma_reward} and the last episode runs to {t} time steps!")
            break


def test(model_name, n_episodes=10):
    mA = Actor()
    mC = Critic()

    mA.load_state_dict(torch.load(f'./preTrained/A{model_name}_actor.pth'))
    mC.load_state_dict(torch.load(f'./preTrained/C{model_name}_critic.pth'))
    render = True

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action, _ = mA.select_action(state)
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
    env_list = ['HalfCheetah-v2', 'InvertedPendulum-v2',
                'Reacher2d-v2', 'Walker2d-v2']
    env = gym.make('')
    env.seed(random_seed)
    torch.manual_seed(random_seed)

    train(model_name, ewma_threshold, lr)
    test()
