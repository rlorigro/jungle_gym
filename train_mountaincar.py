import os
import sys
import numpy

import ctypes

from handlers.FileManager import FileManager
from models.PolicyGradient import Policy, History, Categorical2

from torch.distributions import Categorical
import torch.optim as optim
import torch
import random
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib

from datetime import datetime
import time
import os

import gymnasium as gym
import numpy as np
import argparse

print(torch.__version__)
print(torch.distributed.is_available())
print(rpc.is_available())


def initialize(rank, world_size):
    print("torch.distributed.is_initialized()", torch.distributed.is_initialized())

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    rpc.init_rpc(
        "worker_" + str(rank),
        world_size=world_size,
        rank=rank)

    print("torch.distributed.is_initialized()", torch.distributed.is_initialized())

    if rank != 0:
        rpc.shutdown()


def update_policy(policy: Policy, optimizer, batch):
    for policy_episode,rewards_episode,_ in batch:
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in reversed(rewards_episode):
            R = r + policy.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + float(np.finfo(np.float32).eps))

        # print(rewards)

        # Convert to one big torch tensor for the whole episode
        policy_episode = torch.stack(policy_episode)

        # Calculate loss
        loss = torch.sum(torch.mul(policy_episode, rewards).mul(-1), dim=-1)

        # Update network weights
        loss.backward()

    optimizer.step()
    optimizer.zero_grad()


'''
Fill a list with computed rollouts from environment
'''
def fetch_batch(model, futures, environment, max_pos: float, termination_pos: float, batch_size: int, world_size: int):
    # Reset batch
    batch = list()

    # Launch async fn calls
    for i in range(1, world_size):
        if i in futures:
            if not futures[i].done():
                futures[i].set_result(None)

        futures[i] = rpc.rpc_async(i, producer_function, args=(i, model, environment, max_pos, termination_pos), timeout=0)

    # Check repeatedly for completion of async calls and relaunch any completed ones until batch is full
    done = False
    while not done:
        for i, future in futures.items():
            if not future.done():
                continue

            else:
                batch.append(future.value())

                observed_max_pos = future.value()[2]

                if observed_max_pos > max_pos:
                    max_pos = observed_max_pos
                    print("Max pos is %.3f on thread %d" % (observed_max_pos, i))

                if len(batch) < batch_size:
                    futures[i] = rpc.rpc_async(i, producer_function, args=(i, model, environment, max_pos, termination_pos), timeout=0)
                else:
                    done = True
                    break

    print("Batch complete")

    return batch


def update_termination_position(batch, termination_pos):
    y = numpy.mean([x[2] for x in batch])
    y += abs(0.1*termination_pos)

    print("Updating termination position from %.3f to %.3f" % (termination_pos,y))

    return y


def consumer_function(rank, world_size):
    if rank != 0:
        exit("ERROR: Consumer rank must be 0")

    initialize(rank, world_size)

    env_name = "MountainCar-v0"

    # Initialize environment
    environment = gym.make(env_name)

    max_pos = float(-sys.maxsize)

    observation_space = environment.observation_space
    action_space = environment.action_space
    gamma = 0.99
    dropout_rate = 0.1

    policy = Policy(state_space=observation_space, action_space=action_space, gamma=gamma, dropout_rate=dropout_rate)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4, weight_decay=1e-4)

    batch_size = 8

    termination_pos = -0.2

    futures = dict()

    while True:
        batch = fetch_batch(model=policy, futures=futures, batch_size=batch_size, world_size=world_size, max_pos=max_pos, termination_pos=termination_pos, environment=environment)
        update_policy(policy=policy, optimizer=optimizer, batch=batch)
        termination_pos = update_termination_position(batch=batch, termination_pos=termination_pos)

    rpc.shutdown()


def producer_function(rank, policy,  environment, max_pos, goal_pos):

    policy_episode = list()
    reward_episode = list()

    categorical = Categorical2([environment.action_space.n])
    latest_max_pos = max_pos
    pos = 0

    while True:
        state, info = environment.reset()  # Reset environment and record the starting state
        terminated = False

        for t in range(400):
            action = select_action(policy=policy, state=state, categorical=categorical, policy_episode=policy_episode)

            state, reward, terminated, truncated, info = environment.step(action)
            pos, velocity = state

            terminated = (pos > goal_pos)

            reward_episode.append(reward)

            if terminated:
                break

        if terminated:

            if pos > max_pos:
                latest_max_pos = pos
            return policy_episode, reward_episode, latest_max_pos


def select_action(policy, state, categorical: Categorical2, policy_episode):
    # print("ACTION SELECTION")

    # perform forward calculation on the environment state variables to obtain the probability of each action state
    state = torch.FloatTensor(state)
    action_state = policy.forward(state)
    # print(action_state)

    # Sample based on the output probabilities of the model
    categorical.set_probs(action_state)
    action = categorical.sample()
    # print(choice_distribution)
    # print(action)

    # Add log probability of our chosen action to history
    action_probability = categorical.log_prob(action)

    policy_episode.append(action_probability)

    return action.item()


def main():
    n_processes = 8
    processes = list()

    for i in range(n_processes):
        if i == 0:
            processes.append(mp.Process(target=consumer_function, args=(i, n_processes)))
        else:
            processes.append(mp.Process(target=initialize, args=(i, n_processes)))

        processes[-1].start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
