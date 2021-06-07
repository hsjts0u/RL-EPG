# Spring 2021, Reinforcement Learning
# Team Implementation Project: Expected Policy Gradient

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor network constructed as specified in the experiment details of the EPG paper
    Layer1 100 neurons
    Layer2  50 neurons
    Layer3  25 neurons
    ReLU nonlinearities
    tanh action output
    """
    def __init__(self, observation_space, action_space):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(observation_space, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, action_space)
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.tanh(self.fc4(out))
        return out
    
    
class Critic(nn.Module):
    """
    Critic network constructed as specified in the experiment details of the EPG paper
    Layer1 100 neurons
    Layer2 100 neurons
    ReLU nonlinearities
    """
    def __init__(self, observation_space, action_space):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(observation_space + action_space, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
