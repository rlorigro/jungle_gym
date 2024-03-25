from handlers.FileManager import FileManager
from models.PolicyGradient import Policy, History, Categorical2

import torch.multiprocessing as mp
from torch.distributions import Categorical
import torch.optim as optim
import torch

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib

from datetime import datetime
import time
import os

import gymnasium as gym
import pandas as pd
import numpy as np
import argparse

matplotlib.use('Agg')


def get_timestamp_string():
    time = datetime.now()
    year_to_second = [time.year, time.month, time.day, time.hour, time.second]

    time_string = "-".join(map(str,year_to_second))

    return time_string


def save_model(output_directory, model, episode):
    FileManager.ensure_directory_exists(output_directory)

    timestamp = get_timestamp_string()
    filename = "model_" + str(episode) + "_" + timestamp
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

            if total_sum == 0:
                total_sum = 1

            mean = total_sum/total_count

            return_array[i][j] = mean
    return return_array


def update_policy(policy: Policy, optimizer, history: History, step: bool):
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in reversed(history.reward_episode):
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)

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

    # Update network weights
    loss.backward()

    if step:
        optimizer.step()
        optimizer.zero_grad()
        history.set_new_goal()

    # Save and initialize episode history counters
    history.loss_history.append(loss.data.item())
    history.reward_history.append(np.sum(history.reward_episode))
    history.reset_episode()


def initialize_plot():
    fig = plt.figure(layout="tight", figsize=[12,9])
    gs = GridSpec(nrows=4, ncols=12, figure=fig)

    axes = list()

    axes.append(fig.add_subplot(gs[0,0:6]))   # 0
    axes[-1].set_ylabel("Episode actions")

    axes.append(fig.add_subplot(gs[1,0:6]))   # 1
    axes[-1].set_ylabel("Episode states")

    axes.append(fig.add_subplot(gs[2:,0:5]))  # 2
    axes[-1].set_xlabel("Velocity")
    axes[-1].set_ylabel("Position")

    axes.append(fig.add_subplot(gs[:2,7:]))    # 3
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_ylabel("Max observed position")

    axes.append(fig.add_subplot(gs[2:,7:]))    # 4
    axes[-1].set_xlabel("Episode")
    axes[-1].set_ylabel("Max observed position")

    # plt.ion()

    return fig,axes


def update_plot(fig, axes, history, e, n_steps, start_time, output_dir):
    # Clear and re-label axes 0
    for artist in axes[0].lines + axes[0].collections:
        artist.remove()

    for artist in axes[1].lines + axes[1].collections:
        artist.remove()

    axes[0].plot(history.action_episode, color="C0")
    axes[0].relim()

    axes[1].plot([x[0] for x in history.state_episode], color="C0")
    axes[1].plot([x[1] for x in history.state_episode], color="C1")
    axes[1].relim()

    arrayB = compute_mean(history.action_history)
    m = axes[2].matshow(arrayB, vmin=0, vmax=2, cmap=matplotlib.colormaps["seismic"])
    axes[2].xaxis.set_ticks_position('bottom')

    # add colorbar if it doesn't exist already
    if n_steps == 1:
        fig.colorbar(m, ax=axes[2])

    elapsed = time.time() - start_time
    axes[3].plot(elapsed, history.get_pos_record(), marker='o', color="C0")
    axes[4].plot(e, history.get_pos_record(), marker='o', color="C0")

    output_path = os.path.join(output_dir, "mountaincar_progress_%d.png" % e)
    plt.savefig(output_path, dpi=200)
    history.reset_action_history()


def train(id, output_directory, policy, env_name, pos_goal):
    history = History()

    if pos_goal is not None:
        history.set_pos_record(pos_goal)
        history.set_new_goal()

    start_time = time.time()

    fig, axes = initialize_plot()

    # Hyperparameters
    n_episodes = 10_000_000
    learning_rate = 1e-4
    batch_size = 16

    optimizer = optim.SGD(policy.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

    environment = gym.make(env_name) #, render_mode="human")

    categorical = Categorical2([policy.ouput_size])

    # Rows correspond to reward, action, state

    n_success = 0
    n_steps = 0

    for e in range(n_episodes):
        state, info = environment.reset()  # Reset environment and record the starting state

        truncated = True
        terminated = False
        t = 0

        for t in range(400):
            action = select_action(policy=policy, state=state, history=history, categorical=categorical)

            state, reward, terminated, truncated, info = environment.step(action)

            pos,velocity = state
            # position is clipped to the range [-1.2, 0.6]
            # starting [-0.6 , -0.4]
            # use history for goal post

            terminated = (pos > history.get_pos_goal())
            # terminated = (pos > -0.3)

            history.reward_episode.append(reward)

            # TODO: fix pos record setting
            if terminated or truncated:
                history.set_pos_record(pos)
                if id == 0:
                    print(e, t, n_success, "max pos: %.3f" % history.get_pos_record(), "goal post: %.3f" % history.get_pos_goal(), terminated)

                break

        if terminated:
            n_success += 1
            step = (n_success % batch_size == 0)

            n_steps += step

            if id == 0 and step:
                update_plot(fig=fig, axes=axes, history=history, e=e, n_steps=n_steps, start_time=start_time)

            update_policy(policy=policy, optimizer=optimizer, history=history, step=step)

        else:
            history.reset_episode()

        if e % 10_000 == 0 and id == 0:
            # plt.show()
            # plt.close()
            print('Episode {}\tLast length: {:5d}'.format(e, t))

            save_model(episode=e, output_directory=output_directory, model=policy)


def select_action(policy, state, history: History, categorical: Categorical2):
    # print("ACTION SELECTION")

    # print(state, type(state))
    # print("state:", state.shape)

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
    history.policy_episode.append(action_probability)
    history.action_episode.append(int(action.data.item()))
    history.state_episode.append(state.numpy())

    # print(action_probability)
    # print()

    return action.item()


def run(model_path, pos_goal, output_dir, n_threads):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        exit("ERROR: output dir already exists: " + output_dir)

    env_name = "MountainCar-v0"
    gamma = 0.99

    # Initialize environment
    environment = gym.make(env_name) #, render_mode="human")

    # Access environment/agent variables
    observation_space = environment.observation_space
    action_space = environment.action_space

    policy = Policy(action_space=action_space, state_space=observation_space, dropout_rate=0.2, gamma=gamma)

    if model_path is not None:
        policy.load_state_dict(torch.load(model_path))

    policy.share_memory()

    processes = list()
    for r in range(n_threads):
        p = mp.Process(target=train, args=(r, output_dir, policy, env_name, pos_goal))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    policy.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pos_goal", type=float, required=False, default=None)
    parser.add_argument("--model_path", type=str, required=False, default=None)
    parser.add_argument("--output_dir", type=str, required=False, default=None)
    parser.add_argument("--n_threads", type=int, required=False, default=None)

    args = parser.parse_args()

    run(model_path=args.model_path, pos_goal=args.pos_goal, output_dir=args.output_dir, n_threads=args.n_threads)
