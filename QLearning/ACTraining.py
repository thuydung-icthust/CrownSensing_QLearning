from operator import itruediv
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
import time

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="Mode of training, resuming or testing",
                    type=str, default='train')
parser.add_argument("--filename", help="Path to the checkpoint file",
                    type=str, default='')

args = parser.parse_args()

seed = 99
np.random.seed(seed)
tf.random.set_seed(seed)

acAgent = ActorCritic(param.num_car, dqn_conf.INPUTNUM, param.n_size, 
                        dqn_conf.ACTIONNUM, file_name=args.filename)

inputfile = "input/carname.txt"

# Create header for saving DQN learning file
now = datetime.datetime.now()  # Getting the latest datetime
# Defining header for the save file
header = ["Episode", "Reward", "Running_reward", "Cover_area", "Sharing_factor", "Sent_pkg"]
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv"
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(
        f, encoding='utf-8', index=False, header=True)

total_node, nodes, min_x, max_x, min_y, max_y, radius = read_data(inputfile)
gnb = GnbServer(total_node=total_node)
net = Network(list_node=nodes, num_node=total_node, gnb=gnb, radius=radius,
              step_length=param.cover_time, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)


running_reward = 0
terminate = False

Tmax = param.update_step
To = param.cover_time
Ro = param.cover_radius

idx_start = 0
if args.mode == 'resume' and args.filename != '':
    idx_start = int((args.filename.split('.')[0]).split('_')[-1]) + 1 
s_time = time.time()
n_ep_max = dqn_conf.N_EPISODE
if args.mode == 'test':
    n_ep_max = 1
for episode_i in range(idx_start, n_ep_max):
    state = net.get_state()

    ep_reward = 0
    ep_area = 0
    ep_sharing = 0
    ep_pkgs = 0

    if args.mode == 'test':
        rews = []
        e_areas = []
        e_sharings = []
        e_sents = []
        e_ovls = []
    # start_time = time.time()
    t = 0
    itr = 0
    while t < MAX_SIMULATE_TIME:
        action_probs_history = [[] for i in range(total_node)]
        critic_value_history = [[] for i in range(total_node)]
        actor_rewards_history = [[] for i in range(total_node)]
        with tf.GradientTape() as tape:
            # print(f'gradient taping: {time.time() - start_time}')
            # start_time = time.time()
            delta_t = 0
            # pass down data for each worker
            for i in range(param.num_car):
                acAgent.knowledges[i].update_infos(state[:, :2], state[:, 2:])

            print('[TRAINING...]')
            while delta_t < Tmax:
                state = tf.convert_to_tensor(state, dtype=tf.float32)
                actions = []
                acted_agents = []
                # print(f'convert state: {time.time() - start_time}')
                # start_time = time.time()
                for i in range(param.num_car):
                    if (delta_t + t == acAgent.action_steps[i]):
                        acted_agents.append(i)
                        map_state = acAgent.knowledges[i].build_matrix_state(acAgent.action_steps[i])
                        map_state = tf.convert_to_tensor(map_state, dtype=tf.float32)
                        # print(f'build matrix: {time.time() - start_time}')
                        # start_time = time.time()
                        action_probs, critic_value = acAgent.forward(
                            tf.expand_dims(state[i], 0), tf.expand_dims(map_state, 0), i)
                        # print(f'forward pass: {time.time() - start_time}')
                        # start_time = time.time()
                        critic_value_history[i].append(critic_value[0, 0])
                        action = acAgent.act(action_probs)
                        # print(f'agent i act: {time.time() - start_time}')
                        # start_time = time.time()
                        action_probs_history[i].append(tf.math.log(
                            tf.clip_by_value(action_probs[0, action], 1e-10, 1.0)))
                        actions.append(action)

                        # reset the action_step
                        velocity = (state[i, 2]**2 + state[i, 3]**2)**(1 / 2)
                        if (velocity == 0):
                            new_step = To
                        else:
                            new_step = int(1 / ((1 / To) + (velocity / (2 * Ro))))
                        rmd = int(new_step / param.step_length)
                        acAgent.action_steps[i] = t + delta_t + (1 + rmd) * param.step_length
                        # print(f'update action step: {time.time() - start_time}')
                        # start_time = time.time()
                    else:
                        actions.append(0)   
                # print(f'agent act time: {time.time() - start_time}')
                # start_time = time.time()
                # print(f'action_steps: {acAgent.action_steps}')
                # print(f'actions: {actions}')
                reward, cover_area, overlap_area, sharing_factor, sent_factor = net.step(actions, acted_agents, t, t + delta_t, min(acAgent.action_steps), episode_i, test=True)
                delta_t += min(acAgent.action_steps) - t - delta_t
                # print(f'env step time: {time.time() - start_time}')
                # start_time = time.time()
                
                # print(f'reward: {reward}')
                # print(f'cover_area: {cover_area}')
                avg_reward = 0
                for i in acted_agents:
                    avg_reward += reward[i]
                    actor_rewards_history[i].append(reward[i])

                avg_reward /= len(acted_agents)
                # print(f'rewards: {reward}')

                next_state = net.get_state()
                terminate = net.check_terminate(t + delta_t)

                ep_reward += avg_reward
                ep_area += cover_area
                ep_sharing += sharing_factor
                ep_pkgs += sent_factor
                if args.mode == 'test':
                    rews.append(avg_reward)
                    e_areas.append(cover_area)
                    e_sharings.append(sharing_factor)
                    e_sents.append(sent_factor)
                    e_ovls.append(overlap_area)
                itr += 1
                state = next_state
               
            returns = [[] for i in range(total_node)]
            for i in range(total_node):
                discounted_sum = 0
                for r in actor_rewards_history[i][::-1]:
                    discounted_sum = r + discounted_sum * param.gamma2
                    returns[i].insert(0, discounted_sum)
                # Normalize
                returns[i] = np.array(returns[i])
                returns[i] = (returns[i] - np.mean(returns[i])) / (np.std(returns[i]) + param.eps)
                returns[i] = returns[i].tolist()
            
            # print(f'backprop prepare time: {time.time() - start_time}')
            # start_time = time.time()
            # print(len(action_probs_history))

            # tape.watch(action_probs_history)
            # tape.watch(critic_value_history)

            # Calculating loss values to update our network
            acAgent.backprop(action_probs_history, critic_value_history, returns, tape)

            # Clear the loss and reward history
            del(action_probs_history)
            del(critic_value_history)
            del(actor_rewards_history)
            reset_tracking(net)
            # print(f'backprop time: {time.time() - start_time}')
        t += Tmax
        if terminate:
            break
    
    print(f'finish training {time.time() - s_time}')
    # Update running reward to check condition for solving
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    acAgent.reset_action_step()
    save_data = np.hstack(
        [episode_i + 1, ep_reward, running_reward, ep_area / itr, ep_sharing / itr, ep_pkgs / itr]).reshape(1, 6)

    with open(filename, 'a') as f:
        pd.DataFrame(save_data).to_csv(
            f, encoding='utf-8', index=False, header=False)

    if args.mode == 'test':
        sdata = pd.DataFrame()
        sdata['reward'] = rews
        sdata['cover_area'] = e_areas
        sdata['sharing'] = e_sharings
        sdata['sent'] = e_sents
        sdata['ovl_area'] = e_ovls
        with open('Data/test.csv', 'w') as f:
            sdata.to_csv(
            f, encoding='utf-8', index=False)
    if (episode_i % 10 == 0):
        template = "Episode time: {:.2f} - Running reward: {:.2f} - Episode reward: {:.2f} - Episode {}"
        print(template.format(time.time() - s_time,running_reward, ep_reward, episode_i))

    if (episode_i % SAVE_NETWORK == 0):
        acAgent.save_network(episode_i)
