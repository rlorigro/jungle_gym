import random

import time
import torch
import math
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


import torch
from torch.distributions.utils import probs_to_logits
class Categorical2:
    def __init__(self, probs_shape):
        # NOTE: probs_shape is supposed to be
        #       the shape of probs that will be
        #       produced by policy network
        if len(probs_shape) < 1:
            raise ValueError("`probs_shape` must be at least 1.")
        self.probs_dim = len(probs_shape)
        self.probs_shape = probs_shape
        self._num_events = probs_shape[-1]
        self._batch_shape = probs_shape[:-1] if self.probs_dim > 1 else torch.Size()
        self._event_shape=torch.Size()

    def set_probs_(self, probs):
        self.probs = probs
        self.logits = probs_to_logits(self.probs)

    def set_probs(self, probs):
        self.probs = probs / probs.sum(-1, keepdim=True)
        self.logits = probs_to_logits(self.probs)

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape(sample_shape + self._batch_shape + self._event_shape)

    def log_prob(self, value):
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)


class History:
    def __init__(self):
        # Intra-episode only, reset after each env rollout
        self.state_episode = None
        self.action_episode = None
        self.policy_episode = None
        self.reward_episode = None

        #self.state_average
        # |---------|
        # |  |   |  |
        # n = 3
        #
        # step_size = 0.1
        self.action_history = None
        self.reset_action_history()
        # history.state_episode #position/velocity
        # history.action_episode   #actions
        # weighted average 50 50 3

        # Spanning across episodes
        self.reward_history = list()
        self.loss_history = list()

        self.reset_episode()

        self.policy_cache = list()
        self.reward_cache = list()
        self.max_cache_size = 400

    def update_cache(self, rewards_episode, policy_episode):
        assert(len(self.reward_cache) == len(self.policy_cache))
        assert(len(rewards_episode) == len(policy_episode))

        j = random.randrange(len(rewards_episode))

        if len(self.reward_cache) < self.max_cache_size:
            self.reward_cache.append(rewards_episode[j])
            self.policy_cache.append(policy_episode[j])
        else:
            i = random.randrange(len(self.reward_cache))

            self.reward_cache[i] = rewards_episode[j]
            self.policy_cache[i] = policy_episode[j]

    def pos_index(self,value):
        return math.ceil((value - -1.2) / 0.01)

    def vel_index(self,value):
        return math.ceil((value- -0.07) / 0.001)

    def reset_action_history(self):
        count_c = self.pos_index(0.6) + 1
        count_v = self.vel_index(0.07) + 1
        self.action_history = np.zeros((int(count_c), int(count_v), 3))  # third dimension  position represents a count for the action

    def capture_actions(self):
        if self.state_episode is None or self.action_episode is None :
            return

        for state,action in zip(self.state_episode,self.action_episode):
            pos_p = self.pos_index(state[0])
            pos_v = self.vel_index(state[1])

            pos_p = int(round(pos_p,2))
            pos_v = int(round(pos_v,3))
            self.action_history[pos_p][pos_v][action] += 1

    def reset_episode(self):
        # Episode policy and reward history
        self.capture_actions()
        self.state_episode = list()
        self.action_episode = list()
        self.policy_episode = list()
        self.reward_episode = list()


class Policy(nn.Module):
    def __init__(self, state_space, action_space, dropout_rate, gamma):
        super(Policy, self).__init__()
        self.input_size = state_space.shape[0]
        self.ouput_size = action_space.n

        self.linear1 = nn.Linear(self.input_size, 128, bias=False)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(128, self.ouput_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.gamma = gamma

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x

