"""
Laurent LEQUIEVRE
Research Engineer, CNRS (France)
ISPR - MACCS Team
Institut Pascal UMR6602
laurent.lequievre@uca.fr

Melodie DANIEL
Post-doctorate
ISPR - MACCS Team
Institut Pascal UMR6602
melodie.daniel@sigma-clermont.fr
"""

import numpy as np
import gym
from collections import deque
import random
from mpi4py import MPI
import torch
import torch.nn as nn


# sync_networks across the different cores
def sync_networks(network):
    comm = MPI.COMM_WORLD
    flat_params = _get_flat_params_or_grads(network, mode='params')
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params_or_grads(network, flat_params, mode='params')

def sync_grads(network):
    flat_grads = _get_flat_params_or_grads(network, mode='grads')
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    _set_flat_params_or_grads(network, global_grads, mode='grads')

# get the flat grads or params
def _get_flat_params_or_grads(params, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    # return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])
    flat_list = []
    for param in params:
            tensor = getattr(param, attr)
            if tensor is not None:
                flat_list.append(tensor.cpu().numpy().flatten())
            elif mode == 'grads':
                flat_list.append(np.zeros_like(param.data.cpu().numpy().flatten()))
    return np.concatenate(flat_list)

    

def _set_flat_params_or_grads(params, flat_params, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    # the pointer
    pointer = 0
    # for param in network.parameters():
    #     getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
    #     pointer += param.data.numel()
    for param in params:
        numel = param.data.numel()
        chunk = torch.tensor(flat_params[pointer:pointer + numel]).view_as(param.data)
        if mode == 'params':
            getattr(param, attr).copy_(chunk)
        else:
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
            param.grad.copy_(chunk)
        pointer += numel


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, num_actions, action_space_low, action_space_high, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = num_actions
        self.low          = action_space_low
        self.high         = action_space_high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

    def get_action_orientation_translation_agents(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return [np.clip(action[0] + ou_state, self.low, self.high), np.clip(action[1] + ou_state, self.low, self.high)]
    


class OUNoiseTorch:
    def __init__(self, num_agents, num_actions, action_space_low, action_space_high,
                 mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000, device="cpu"):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.n_agents     = num_agents
        self.action_dim   = num_actions
        self.low          = torch.tensor(action_space_low, device=device)
        self.high         = torch.tensor(action_space_high, device=device)
        self.device       = device
        self.reset()
        
    def reset(self):
        # State is now of shape (n_agents, action_dim)
        self.state = torch.ones(self.n_agents, self.action_dim, device=self.device) * self.mu
        
    def evolve_state(self):
        # Vectorized noise evolution for all agents
        dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn(self.n_agents, self.action_dim, device=self.device)
        self.state = self.state + dx
        return self.state
    
    def get_action(self, action, t=0):
        # Evolve noise for all agents
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        # Adding the noise and clipping for all agents at once
        return torch.clamp(action + ou_state, self.low, self.high)


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
        

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(int(done))
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
    


from collections import deque

class MovingAverageReward:
    def __init__(self, window_size=100, device="cpu"):
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)
        self.device = device

    def add_reward(self, reward):
        """
        Add a new reward and update the moving average.
        """
        if not torch.is_tensor(reward):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.rewards.append(reward)

    def get_average(self):
        """
        Get the current moving average of the stored rewards.
        """
        if not self.rewards:
            return torch.tensor(0.0, device=self.device)
        return torch.mean(torch.stack(list(self.rewards)))


        
