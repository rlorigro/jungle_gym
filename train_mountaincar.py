import os
import argparse
import sys
import numpy
from copy import deepcopy

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


# print(torch.__version__)
# print(torch.distributed.is_available())
# print(rpc.is_available())


def initialize(rank, world_size):
    # print("torch.distributed.is_initialized()", torch.distributed.is_initialized())

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    rpc.init_rpc(
        "worker_" + str(rank),
        world_size=world_size,
        rank=rank)

    # print("torch.distributed.is_initialized()", torch.distributed.is_initialized())

    if rank != 0:
        rpc.shutdown()


def update_policy(policy: Policy, optimizer, batch):
    for policy_episode, rewards_episode, _ in batch:
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
        del rewards
        # Update network weights
        loss.backward()

    optimizer.step()
    optimizer.zero_grad()


def get_timestamp_string():
    time = datetime.now()
    year_to_second = [time.year, time.month, time.day, time.hour, time.second]
    time_string = "-".join(map(str, year_to_second))
    return time_string


def save_model(output_directory, model, i):
    FileManager.ensure_directory_exists(output_directory)
    timestamp = get_timestamp_string()
    filename = "model_" + str(i) + "_" + timestamp
    path = os.path.join(output_directory, filename)
    print("SAVING MODEL:", path)
    torch.save(model.state_dict(), path)


'''
Fill a list with computed rollouts from environment
'''


def fetch_batch(model, futures, environment_name, max_pos: float, termination_pos: float, batch_size: int,
                world_size: int):
    # Reset batch
    batch = list()

    # Launch async fn calls
    for i in range(1, world_size):
        if i in futures:
            if not futures[i].done():
                pass
                # print(i, "done maybe after set result?")
            else:
                futures[i] = rpc.rpc_async(i, producer_function, args=(i, model, environment_name, termination_pos),
                                           timeout=0)
        else:
            futures[i] = rpc.rpc_async(i, producer_function, args=(i, model, environment_name, termination_pos),
                                       timeout=0)

    # Check repeatedly for completion of async calls and relaunch any completed ones until batch is full
    done = False

    """
     P |  1    |  2     |  3     |
    ------------------------------
     M | 0.2  |  0.25  |  0.25  |
    ------------------------------
    OM | 0.23  |  0.28  |  0.28  |

    """

    while not done:
        for i, future in futures.items():
            if not future.done():
                continue

            else:
                batch.append(future.value())
                observed_max_pos = future.value()[2]

                if observed_max_pos > max_pos:
                    max_pos = observed_max_pos
                    print("Max pos is %.3f on thread %d (higher than max)" % (observed_max_pos, i))
                else:
                    print("Max pos is %.3f on thread %d " % (observed_max_pos, i))

                if len(batch) < batch_size:
                    futures[i] = rpc.rpc_async(i, producer_function, args=(i, model, environment_name, termination_pos),
                                               timeout=0)
                else:
                    done = True
                    break

    print("Max pos is %.3f             (batch max)" % (max_pos))
    print("Batch complete")

    return batch, max_pos


def update_termination_position(batch):
    y = numpy.mean([x[2] for x in batch])
    y += abs(0.1 * y)

    y = min(0.55, y)

    return y


def consumer_function(rank, world_size, output_directory, termination_pos, model_path, batch_size, learning_rate):
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
    dropout_rate = 0.05

    policy = Policy(state_space=observation_space, action_space=action_space, gamma=gamma, dropout_rate=dropout_rate)
    if model_path is not None:
        policy.load_state_dict(torch.load(model_path))
        policy.eval()
    policy.share_memory()

    optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate, weight_decay=1e-4)

    futures = dict()

    try:
        i = 0
        while True:
            batch, max_pos = fetch_batch(model=policy, futures=futures, batch_size=batch_size, world_size=world_size,
                                         max_pos=max_pos, termination_pos=termination_pos, environment_name=env_name)
            update_policy(policy=policy, optimizer=optimizer, batch=batch)
            save_model(i=i, output_directory=output_directory, model=policy)
            termination_pos = update_termination_position(batch=batch)

            i += 1

    except Exception as e:
        rpc.shutdown()
        raise e


def producer_function(rank, policy, environment_name, goal_pos):
    environment = gym.make(environment_name)
    categorical = Categorical2([environment.action_space.n])

    observed_max_pos = -sys.maxsize
    pos = 0

    while True:
        state, info = environment.reset()  # Reset environment and record the starting state
        terminated = False

        policy_episode = list()
        reward_episode = list()

        for t in range(400):
            action = select_action(policy=policy, state=state, categorical=categorical, policy_episode=policy_episode)

            state, reward, terminated, truncated, info = environment.step(action)
            pos, velocity = state

            terminated = (pos > goal_pos)

            reward_episode.append(reward)

            if terminated:
                break

        if terminated:
            if pos > observed_max_pos:
                observed_max_pos = pos

            del environment
            return policy_episode, reward_episode, observed_max_pos


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


def train_model(n_processes, output_dir, termination_pos, model_path, batch_size, learning_rate):
    ##add something here to load model

    if (model_path is None):
        print("no model path provided")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processes = list()

    for i in range(n_processes):
        if i == 0:
            processes.append(mp.Process(target=consumer_function, args=(
            i, n_processes, output_dir, termination_pos, model_path, batch_size, learning_rate)))
        else:
            processes.append(mp.Process(target=initialize, args=(i, n_processes)))

        processes[-1].start()

    for p in processes:
        p.join()


def run_model(model_path):
    if (model_path is None):
        print("no model path provided")
        return

    environment = gym.make("MountainCar-v0", render_mode="human")
    observation, info = environment.reset()

    observation_space = environment.observation_space
    action_space = environment.action_space
    gamma = 0.99
    dropout_rate = 0.1

    policy = Policy(state_space=observation_space, action_space=action_space, gamma=gamma, dropout_rate=dropout_rate)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    for _ in range(1000):
        action = environment.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = environment.step(action)

        if terminated or truncated:
            observation, info = environment.reset()

    environment.close()
    return None


def main(run_mode, n_processes, output_dir, termination_pos, model_path, batch_size, learning_rate):
    if run_mode:
        run_model(model_path)
    else:
        train_model(n_processes, output_dir, termination_pos, model_path, batch_size, learning_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_processes", type=int, required=False, default=8)
    parser.add_argument("--batch_size", type=int, required=False, default=8)
    parser.add_argument("--learning_rate", type=float, required=False, default=1e-3)
    parser.add_argument("--output_dir", type=str, required=False, default="output")
    parser.add_argument("--termination_pos", type=float, required=False, default=-0.2)
    parser.add_argument("--run_mode", action=argparse.BooleanOptionalAction)
    parser.add_argument("--model_path", type=str, required=False, default=None)

    args = parser.parse_args()
    main(
        n_processes=args.n_processes,
        output_dir=args.output_dir,
        termination_pos=args.termination_pos,
        run_mode=args.run_mode,
        model_path=args.model_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
