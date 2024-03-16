import multiprocessing
import random

import torch.multiprocessing as mp
from handlers.FileManager import FileManager
from models.PolicyGradient import Policy, History
from torch.distributions import Categorical
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import gymnasium as gym
import sys
import os


def get_timestamp_string():
    time = datetime.now()
    year_to_second = [time.year, time.month, time.day, time.hour, time.second]

    time_string = "-".join(map(str,year_to_second))

    return time_string


def save_model(output_directory, model, id):
    FileManager.ensure_directory_exists(output_directory)

    timestamp = get_timestamp_string()
    filename = "model_" + str(id) + "_" + timestamp
    path = os.path.join(output_directory, filename)

    print("SAVING MODEL:", path)
    torch.save(model.state_dict(), path)


def plot_results(n_episodes, history: History):
    window = int(n_episodes / 20)

    print(len(history.reward_history))
    print(history.reward_history)

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])

    rolling_mean = pd.Series(history.reward_history).rolling(window).mean()

    std = pd.Series(history.reward_history).rolling(window).std()

    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(history.reward_history)), rolling_mean - std, rolling_mean + std, color='orange', alpha=0.2)
    ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')

    ax2.plot(history.reward_history)
    ax2.set_title('Episode Length')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')

    fig.tight_layout(pad=2)
    plt.show()




    # fig.savefig('results.png')


def compute_mean(array):
    array_size = array.shape
    dim_x = array_size[0]
    dim_y = array_size[1]

    return_array = np.zeros((dim_x,dim_y))
    for i in range(dim_x):
        for j in range(dim_y):
            count_0 = array[i][j][0]
            count_1 = array[i][j][1]
            count_2 = array[i][j][2]

            total_count = count_0+count_1+count_2

            total_sum = count_1 + count_2*2

            if total_count == 0:
                total_count = 1

            mean = total_sum/total_count

            return_array[i][j] = mean
    return return_array


def update_policy(policy: Policy, optimizer, history: History, step: bool, axes, plot: bool):
    R = 0
    rewards = []

    # print("len of reward episode", len(policy.reward_episode))

    # print(policy.reward_episode)

    # print(len(policy.reward_episode))

    # Discount future rewards back to the present using gamma
    for r in reversed(history.reward_episode):
        R = r + policy.gamma * R
        # R = r
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)

    # print("rewards")
    # print(rewards)

    # Normalize rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + float(np.finfo(np.float32).eps))

    # print(rewards)

    # Convert to one big torch tensor for the whole episode
    policy_episode = torch.stack(history.policy_episode)

    # Calculate loss
    loss = torch.sum(torch.mul(policy_episode, rewards).mul(-1), dim=-1)

    # if history.policy_cache is not None and len(history.policy_cache) > 0:
    #     replay_policy = torch.stack(history.policy_cache)
    #     reward_cache = torch.stack(history.reward_cache)
    #     replay_loss = torch.sum(torch.mul(replay_policy, reward_cache).mul(-1), dim=-1)
    #     replay_loss.backward(retain_graph=True)
    #     print(replay_loss)
    #
    # history.update_cache(rewards_episode=rewards, policy_episode=policy_episode)

    print(loss)

    # print(policy_history)
    # print(p.shape)

    if plot:
        axes[0].plot(torch.exp(rewards.data))
        axes[1].plot(history.action_episode)
        axes[2].plot([x[0] for x in history.state_episode])
        axes[2].plot([x[1] for x in history.state_episode])

        arrayB = compute_mean(history.action_history)
        a = np.diag(range(15))
        axes[3].matshow(arrayB)
        history.reset_action_history()
        plt.show()
        plt.pause(1e-10)
        axes[0].cla()
        axes[1].cla()
        axes[2].cla()

    # sys.stdin.readline()

    # Update network weights

    loss.backward()

    if step:
        optimizer.step()
        optimizer.zero_grad()

    # Save and initialize episode history counters
    history.loss_history.append(loss.data.item())
    history.reward_history.append(np.sum(history.reward_episode))
    history.reset_episode()

    # if e % 50 == 0:
    #     plt.plot(rewards.data.numpy())
    #     # print(loss.data.numpy())
    #     plt.show()
    #     plt.close()


def train(id, output_directory, policy, env_name, render=False):
    history = History()

    # Hyperparameters
    n_episodes = 15000
    learning_rate = 1e-4

    optimizer = optim.SGD(policy.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

    environment = gym.make(env_name) #, render_mode="human")

    # Rows correspond to reward, action, state
    fig, axes = plt.subplots(nrows=4, height_ratios=[1, 1, 1, 3])
    plt.ion()

    n_success = 0

    for e in range(n_episodes):
        state, info = environment.reset()  # Reset environment and record the starting state

        time = 0
        done = False

        max_pos = -float(sys.maxsize)
        max_velocity = -float(sys.maxsize)

        truncated = True
        terminated = False

        # run a simulation for up to 1000 time steps
        for time in range(400):
            if render:
                environment.render()

            # print("TIME: ", time)
            action = select_action(policy=policy, state=state, history=history)

            state, reward, terminated, truncated, info = environment.step(action)
            # print("reward", reward)
            # print(state)

            pos = state[0]
            velocity = state[1]

            # OVERRIDE terminated FOR EASIER GOAL
            terminated = (pos > -0.25)

            if pos > max_pos:
                max_pos = pos
                i_max_pos = time

            if velocity > max_velocity:
                max_velocity = velocity

            # Save reward
            history.reward_episode.append(reward)

            if terminated or truncated:
                if id == 0:
                    print(e, time, n_success, "max pos: %.3f" % max_pos, "max vel: %.3f" % max_velocity, terminated)

                break

        if terminated:
            n_success += 1
            update_policy(policy=policy, optimizer=optimizer, history=history, axes=axes, step=(n_success%25==0), plot=((n_success%50==0) and (id==0)))
        else:
            history.reset_episode()

        if e % 4000 == 0:
            # plt.show()
            # plt.close()
            print('Episode {}\tLast length: {:5d}'.format(e, time))

            save_model(id=id, output_directory=output_directory, model=policy)

    plt.ioff()

    if id == 0:
        plot_results(n_episodes=n_episodes, history=history)


def select_action(policy, state, history: History):
    # print("ACTION SELECTION")

    # print(state, type(state))
    # print("state:", state.shape)

    # perform forward calculation on the environment state variables to obtain the probability of each action state
    state = torch.FloatTensor(state)
    action_state = policy.forward(state)
    # print(action_state)

    # Sample based on the output probabilities of the model
    choice_distribution = Categorical(action_state)
    action = choice_distribution.sample()
    # print(choice_distribution)
    # print(action)

    # Add log probability of our chosen action to history
    action_probability = choice_distribution.log_prob(action)
    history.policy_episode.append(action_probability)
    history.action_episode.append(int(action.data.item()))
    history.state_episode.append(state.numpy())

    # print(action_probability)
    # print()

    return action.item()


def test(policy, environment, render=False):
    history = History()

    while True:
        state, info = environment.reset()  # Reset environment and record the starting state

        # run a simulation for up to 1000 time steps
        for time in range(1000):
            if render:
                environment.render()

            # print("TIME: ", time)
            action = select_action(policy=policy, state=state, history=history)

            state, reward, terminated, truncated, info = environment.step(action)

            # Save reward
            history.reward_episode.append(reward)

            if terminated or truncated:
                print(time)
                environment.reset()
                break


def run():
    render = False

    output_directory = "output"

    # gym.envs.register(
    #     id='CartPole-v666',
    #     entry_point='gym.envs.classic_control:CartPoleEnv',
    #     max_episode_steps=5000,
    #     reward_threshold=5000.0
    # )

    env_name = "MountainCar-v0"
    gamma = 0.99

    # Initialize environment

    environment = gym.make(env_name) #, render_mode="human")

    # Access environment/agent variables
    observation_space = environment.observation_space
    action_space = environment.action_space

    n_processes = 32

    policy = Policy(action_space=action_space, state_space=observation_space, dropout_rate=0.2, gamma=gamma)
    policy.share_memory()

    processes = list()
    for r in range(n_processes):
        p = mp.Process(target=train, args=(r, output_directory, policy, env_name, render))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # train(output_directory=output_directory, environment=environment, policy=policy, render=render)

    policy.eval()

    environment = gym.make(env_name)
    # environment = gym.make(env_name, render_mode="human")

    test(policy, environment, True)


if __name__ == "__main__":
    run()
