import random

import torch
import math
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


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
        # print("what the fuck is this ", self.action_episode)
        # [array([-0.56473184, 0.], dtype=float32), array([-0.5634241, 0.00130772]
        if self.state_episode is None or self.action_episode is None :
            return

        for state,action in zip(self.state_episode,self.action_episode):
            pos_p = self.pos_index(state[0])
            pos_v = self.vel_index(state[1])

            pos_p = int(round(pos_p,2))
            pos_v = int(round(pos_v,3))
            self.action_history[pos_p][pos_v][action] += 1
            """    self.action_history[pos_p][pos_v][action] += 1
    ~~~~~~~~~~~~~~~~~~~^^^^^^^
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices"""

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

