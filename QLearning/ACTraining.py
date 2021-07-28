from config import *
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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="Mode of training or resuming",
                    type=str, default='train')
parser.add_argument("--filename", help="Path to the checkpoint file",
                    type=str, default='')

args = parser.parse_args()

seed = 99
np.random.seed(seed)
tf.random.set_seed(seed)

acAgent = ActorCritic(param.num_car, dqn_conf.INPUTNUM, param.n_size, dqn_conf.ACTIONNUM, file_name=args.filename)

inputfile = "input/carname.txt"

# Create header for saving DQN learning file
now = datetime.datetime.now()  # Getting the latest datetime
# Defining header for the save file
header = ["Episode", "Reward", "Running_reward", "Cover_area", "Sharing_factor", "Sent_pkg"]
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv"
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(
        f, encoding='utf-8', index=False, header=True)

total_node, nodes, min_x, max_x, min_y, max_y = read_data(inputfile)
gnb = GnbServer(total_node=total_node)
net = Network(list_node=nodes, num_node=total_node, gnb=gnb,
              step_length=param.cover_time, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)

action_probs_history = []
critic_value_history = []
actor_rewards_history = []
running_reward = 0
terminate = False

Tmax = param.update_step
To = param.cover_time
Ro = param.cover_radius

idx_start = 0
if args.mode == 'resume' and args.filename != '':
    idx_start = int((args.filename.split('.')[0]).split('_')[-1]) + 1 

for episode_i in range(idx_start, dqn_conf.N_EPISODE):
    state = net.get_state()

    ep_reward = 0
    ep_area = 0
    ep_sharing = 0
    ep_pkgs = 0

    for step in range(0, dqn_conf.MAX_STEP):
        is_train = False
        if (step * param.step_length) % Tmax == 0:
            # pass down data for each worker
            for i in range(param.num_car):
                acAgent.knowledges[i].update_infos(state[:, :2], state[:, 2:])
            is_train = True

        if is_train:
            print('[TRAINING...]')
            with tf.GradientTape() as tape:
                state = tf.convert_to_tensor(state, dtype=tf.float32)
                actions = []
                acted_agents = []
                for i in range(param.num_car):
                    if ((step * param.step_length) % acAgent.action_steps[i] == 0):
                        acted_agents.append(i)
                        map_state = acAgent.knowledges[i].build_matrix_state(acAgent.action_steps[i])
                        map_state = tf.convert_to_tensor(map_state, dtype=tf.float32)
                        action_probs, critic_value = acAgent.forward(
                            tf.expand_dims(state[i], 0), tf.expand_dims(map_state, 0), i)

                        critic_value_history.append(critic_value[0, 0])
                        action = acAgent.act(action_probs)
                        action_probs_history.append(tf.math.log(
                            tf.clip_by_value(action_probs[0, action], 1e-10, 1.0)))
                        actions.append(action)

                        # reset the action_step
                        velocity = (state[i, 2]**2 + state[i, 3]**2)**(1 / 2)
                        if (velocity == 0):
                            new_step = To
                        else:
                            new_step = int(1 / ((1 / To) + (velocity / (2 * Ro))))

                        rmd = int(new_step / param.step_length)
                        if rmd == 0:
                            acAgent.action_steps[i] = param.step_length
                        else:
                            acAgent.action_steps[i] = rmd * param.step_length
                    else:
                        actions.append(0)

                print(f'action_steps: {acAgent.action_steps}')
                print(f'actions: {actions}')
                if (len(acted_agents) != 0):
                    reward, cover_area, sharing_factor, sent_factor = net.step(actions, step, episode_i, test=True)
                    # print(f'reward: {reward}')
                    # print(f'cover_area: {cover_area}')

                    avg_reward = 0
                    for i in acted_agents:
                        actor_rewards_history.append(reward[i])
                        avg_reward += reward[i]

                    print(f'rewards: {reward}')

                    avg_reward /= len(acted_agents)
                    next_state = net.get_state()
                    terminate = net.check_terminate(step)

                    ep_reward += avg_reward
                    ep_area += cover_area
                    ep_sharing += sharing_factor
                    ep_pkgs += sent_factor

                    state = next_state
                else:
                    net.step(actions, step, episode_i, test=False)
                    next_state = net.get_state()
                    terminate = net.check_terminate(step)
                    state = next_state
                    continue
                    # # if no worker work, random a forward pass just to activate gradient tape
                    # map_state = np.random.rand(param.n_size, param.n_size)
                    # map_state = tf.convert_to_tensor(map_state, dtype=tf.float32)
                    # _, _ = acAgent.forward(
                    #     tf.expand_dims(state[0], 0), tf.expand_dims(map_state, 0), 0)

                if ((step * param.step_length) % Tmax == 0):
                    # train the target model
                    returns = []
                    discounted_sum = 0

                    for r in actor_rewards_history[::-1]:
                        discounted_sum = r + discounted_sum * param.gamma2
                        returns.insert(0, discounted_sum)

                    # Normalize
                    returns = np.array(returns)
                    returns = (returns - np.mean(returns)) / (np.std(returns) + param.eps)
                    returns = returns.tolist()

                    # print(len(action_probs_history))

                    # tape.watch(action_probs_history)
                    # tape.watch(critic_value_history)

                    # Calculating loss values to update our network
                    history = zip(action_probs_history, critic_value_history, returns)
                    acAgent.backprop(history, tape)
                    # pass weight to all worker modelss
                    # acAgent.update_weights()

                    # Clear the loss and reward history
                    action_probs_history.clear()
                    critic_value_history.clear()
                    actor_rewards_history.clear()
                    reset_tracking(net)

        else:
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            actions = []
            acted_agents = []
            for i in range(param.num_car):
                if ((step * param.step_length) % acAgent.action_steps[i] == 0):
                    acted_agents.append(i)
                    map_state = acAgent.knowledges[i].build_matrix_state(acAgent.action_steps[i])
                    map_state = tf.convert_to_tensor(map_state, dtype=tf.float32)
                    action_probs, critic_value = acAgent.forward(
                        tf.expand_dims(state[i], 0), tf.expand_dims(map_state, 0), i)

                    critic_value_history.append(critic_value[0, 0])
                    action = acAgent.act(action_probs)
                    action_probs_history.append(tf.math.log(
                        tf.clip_by_value(action_probs[0, action], 1e-10, 1.0)))
                    actions.append(action)

                    # reset the action_step
                    velocity = (state[i, 2]**2 + state[i, 3]**2)**(1 / 2)
                    if (velocity == 0):
                        new_step = To
                    else:
                        new_step = int(1 / ((1 / To) + (velocity / (2 * Ro))))

                    rmd = int(new_step / param.step_length)
                    if rmd == 0:
                        acAgent.action_steps[i] = param.step_length
                    else:
                        acAgent.action_steps[i] = rmd * param.step_length
                else:
                    actions.append(0)

            print(f'action_steps: {acAgent.action_steps}')
            print(f'actions: {actions}')
            if (len(acted_agents) != 0):
                reward, cover_area, sharing_factor, sent_factor = net.step(actions, step, episode_i, test=True)
                # print(f'reward: {reward}')
                # print(f'cover_area: {cover_area}')

                avg_reward = 0
                for i in acted_agents:
                    actor_rewards_history.append(reward[i])
                    avg_reward += reward[i]

                print(f'rewards: {reward}')

                avg_reward /= len(acted_agents)
                next_state = net.get_state()
                terminate = net.check_terminate(step)

                ep_reward += avg_reward
                ep_area += cover_area
                ep_sharing += sharing_factor
                ep_pkgs += sent_factor

                state = next_state
            else:
                net.step(actions, step, episode_i, test=False)
                next_state = net.get_state()
                terminate = net.check_terminate(step)
                state = next_state

    if terminate:
        break

    # Update running reward to check condition for solving
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

    # Clear the loss and reward history
    action_probs_history.clear()
    critic_value_history.clear()
    actor_rewards_history.clear()

    save_data = np.hstack(
        [episode_i + 1, ep_reward, running_reward, ep_area / MAX_STEP, ep_sharing / MAX_STEP, ep_pkgs / MAX_STEP]).reshape(1, 6)

    with open(filename, 'a') as f:
        pd.DataFrame(save_data).to_csv(
            f, encoding='utf-8', index=False, header=False)

    if (episode_i % 10 == 0):
        template = "Running reward: {:.2f} - Episode reward: {:.2f} - Episode {}"
        print(template.format(running_reward, ep_reward, episode_i))

    if (episode_i % SAVE_NETWORK == 0):
        acAgent.save_network(episode_i)
