import torch
import torch.nn as nn
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

        # Spanning across episodes
        self.reward_history = list()
        self.loss_history = list()

        self.reset_episode()

    def reset_episode(self):
        # Episode policy and reward history
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

