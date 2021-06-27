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
from memory import SequentialMemory
from graphing import learning_curve

LR = 0.001
DISCOUNT = 0.99
SIGMA = 0.5
C = 1.0
EIGEN_CLIP = 1

REPLAY_BUFFER_CAP = 100_000
IGNORE_STEP = 10_000

TAU = 0.01

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
            target_param.data * (1.0 - tau) + param.data * tau
        )

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
        self.target_actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.actor_optim = Adam(self.actor.parameters(), lr=LR)
        self.critic_optim = Adam(self.critic.parameters(), lr=LR)
        self.memory = SequentialMemory(limit=REPLAY_BUFFER_CAP, window_length=1)
        
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
    
    def action_mean(self, state):
        return self.actor(to_tensor(state)) * self.action_lim
    
    def select_action(self, mean, covaraince):
        act = MultivariateNormal(mean.double(), covaraince).sample().float()
        act = torch.clamp(act, -self.action_lim, self.action_lim)
        return act
    
    def expm(self, m):
        """ this function can only handle symmetric matrix in numpy """
        with torch.no_grad():
            L, V = torch.symeig(m, eigenvectors=True)
            L = torch.clamp_max(L, EIGEN_CLIP)
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
        # step = nd.step_generators.one_step
        def Qsu(act):
            return self.critic(torch.cat((state_tensor, torch.tensor(act, dtype=torch.float32)), dim=-1)).detach().numpy().astype(np.float64)

        # def Qsu(act):
        #    return self.critic(torch.cat((to_tensor(state), to_tensor(act).unsqueeze(0), dim=-1)).squeeze(0).detach().numpy()
        
        ############################################################
        
        acts = MultivariateNormal(mu.repeat(100, 1), 0.001 * torch.eye(self.action_dim)).sample()
        qs = self.critic(torch.cat((state_tensor.repeat(100, 1), acts), dim=-1))

        sample_Q = qs.reshape(100, 1)
        sample_action = torch.transpose(acts, 0, 1)
        hessian_list = []
        
        for i in range(self.action_dim):
            A = torch.stack([torch.ones(sample_action[i].shape), sample_action[i], torch.square(sample_action[i])], dim=1)
            X, _ = torch.lstsq(sample_Q, A)
            hessian_list.append(X[2] * 2)
        H = torch.diag(torch.cat(hessian_list, dim=0))
        
        ############################################################

        mup = mu.detach().numpy().astype(np.float64)
        # H = nd.Hessian(Qsu)(mup)
        cov = SIGMA * SIGMA * self.expm(C * H)
        # cov = SIGMA * SIGMA * self.expm(C * to_tensor(H, numpy_dtype=np.float64))
        return cov + 0.0001 * torch.eye(self.action_dim, dtype=torch.float64)
        
    def train(self, env, step_limit):
                
        gamma_acc = 1.0
        episode_reward = 0.0
        episode = 1
        episode_step = 0
        
        learn_data = []
        
        loss = nn.MSELoss()
        
        if IGNORE_STEP > 0:
            print("Ignoring steps...")
            obs = env.reset()
            for step in range(IGNORE_STEP):
                mu = self.action_mean(obs)
                action = self.select_action(mu, self.get_covariance(obs, mu)).detach().numpy()
                new_obs, reward, done, _ = env.step(action)
                self.memory.append(obs, action, reward, done)
                obs = deepcopy(new_obs)
                if done:
                    obs = env.reset()
            print(f"Ignored {IGNORE_STEP} steps")
        
        obs = env.reset()
        for step in range(step_limit):
            episode_step += 1
            
            # observ
            mu = self.action_mean(obs)
            action = self.select_action(mu, self.get_covariance(obs, mu)).detach().numpy()
            
            new_obs, reward, done, _ = env.step(action)
            self.memory.append(obs, action, reward, done)
            obs = deepcopy(new_obs)
                    
            # end observ
            
            state_batch, action_batch, reward_batch, next_state_batch, \
                terminal_batch = self.memory.sample_and_split(64)
            ## 
            next_q_values = self.target_critic(
                torch.cat((
                    to_tensor(next_state_batch),
                    self.target_actor(to_tensor(next_state_batch)).detach()
                ), dim=1)
            ).detach()

            target_q_batch = to_tensor(reward_batch) + \
                DISCOUNT * \
                to_tensor(terminal_batch.astype(np.float32))*next_q_values


            q_batch = self.critic(torch.cat(
                (to_tensor(state_batch), to_tensor(action_batch)), dim=1))

            value_loss = loss(q_batch, target_q_batch)
            self.critic_optim.zero_grad()
            value_loss.backward()
            self.critic_optim.step()

            policy_loss = -self.critic(torch.cat(
                (to_tensor(state_batch), self.actor(to_tensor(state_batch))),
                dim=1))

            policy_loss = policy_loss.mean()
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
            
            soft_update(self.target_actor, self.actor, TAU)
            soft_update(self.target_critic, self.critic, TAU)

            ##
            
            episode_reward += reward
            
            if done:
                
                print(f'step {step} episode {episode} reward {episode_reward} in {episode_step} steps')
                learn_data.append((step, episode_reward))
                episode += 1
                episode_reward = 0
                episode_step = 0
                obs = env.reset()
        
        return learn_data


def save_learning_curve(data, file_name):
    """ data should be list of (step, episode_rewards) """
    f = open(file_name, "a")
    for step, reward in data:
        f.write(f"{step} {reward}\n")
    f.write('0 0')
    f.close()


if __name__ == '__main__':
    env_name = 'InvertedPendulum-v2'
    env = gym.make(env_name)
    for _ in range(20):
        agent = epg(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
        data = agent.train(env, 100_000)
        save_learning_curve(data, f"epg/epg_rb_t_lstsq_{env_name}.txt")
