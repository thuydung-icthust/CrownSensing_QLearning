from config import SAVE_NETWORK
import config as dqn_conf
from ACModel import ActorCritic
from Network import Network
from GnbServer import GnbServer
from getdata import read_data
from Network_Method import *
import Parameter as param

import tensorflow as tf
import pandas as pd
import datetime
import numpy as np


seed = 99
np.random.seed(seed)
tf.random.set_seed(seed)


acAgent = ActorCritic(param.num_car, dqn_conf.INPUTNUM, dqn_conf.ACTIONNUM)

inputfile = "input/carname.txt"

# Create header for saving DQN learning file
now = datetime.datetime.now()  # Getting the latest datetime
# Defining header for the save file
header = ["Step", "Reward", "Running_reward", "Cover_area", "Sharing_factor", "Sent_pkg"]
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv"
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(
        f, encoding='utf-8', index=False, header=True)

total_node, nodes, min_x, max_x, min_y, max_y = read_data(inputfile)
print(total_node, min_x, max_x, min_y, max_y)
gnb = GnbServer(total_node=total_node)
net = Network(list_node=nodes, num_node=total_node, gnb=gnb,
              step_length=param.cover_time, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)

action_probs_history = []
critic_value_history = []
actor_rewards_history = []
running_reward = 0
terminate = False

for episode_i in range(0, dqn_conf.N_EPISODE):
    state = net.get_state()
    ep_reward = 0
    ep_area = 0
    ep_sharing = 0
    ep_pkgs = 0
    with tf.GradientTape() as tape:
        for step in range(1, dqn_conf.MAX_STEP):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            print(f'state: {state}')
            actions_probs, critic_value = acAgent.forward(state)
            critic_value_history.append([critic_value[0, i] for i in range(param.num_car)])

            actions = acAgent.act(actions_probs)
            print(f'action: {actions}')

            actions_log = []
            for i in range(param.num_car):
                actions_log.append(tf.math.log(tf.clip_by_value(actions_probs[i][0, actions[i]], 1e-10, 1.0)))
            action_probs_history.append(actions_log)

            reward, cover_area, sharing_factor, sent_factor = net.step(actions, step, episode_i, test=True)
            print(f'reward: {reward}')
            avg_reward = np.average(reward)
            next_state = net.get_state()
            terminate = net.check_terminate(step)

            actor_rewards_history.append(reward)

            ep_reward += avg_reward
            ep_area += cover_area
            ep_sharing += sharing_factor
            ep_pkgs += sent_factor

            state = next_state
            reset_tracking(net, step)

            if terminate:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        for i in range(param.num_car):
            discounted_sum = 0
            actor_i_reward = actor_rewards_history[i]
            returns_i = []
            for r in actor_i_reward[::-1]:
                discounted_sum = r + param.gamma2 * discounted_sum
                returns_i.insert(0, discounted_sum)

            # Normalize
            returns_i = np.array(returns_i)
            returns_i = (returns_i - np.mean(returns_i)) / (np.std(returns_i) + param.eps)
            returns_i = returns_i.tolist()

            returns.append(returns_i)

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        acAgent.backprop(history, tape)

        acAgent.update_epsilon()

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        actor_rewards_history.clear()

    save_data = np.hstack(
        [episode_i + 1, ep_reward, running_reward, ep_area, ep_sharing, ep_pkgs]).reshape(1, 6)

    with open(filename, 'a') as f:
        pd.DataFrame(save_data).to_csv(
            f, encoding='utf-8', index=False, header=False)

    if (episode_i % 10 == 0):
        template = "Running reward: {:.2f} - Episode reward: {:.2f} - Episode {}"
        print(template.format(running_reward, ep_reward, episode_i))

    if (episode_i % SAVE_NETWORK == 0):
        acAgent.save_network(episode_i)
