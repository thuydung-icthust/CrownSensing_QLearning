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


seed = 99
np.random.seed(seed)
tf.random.set_seed(seed)


acAgent = ActorCritic(param.num_car, dqn_conf.INPUTNUM, dqn_conf.ACTIONNUM, epsilon=0)
acAgent.load_model('checkpoint/checkpoint_ac/ac_ep_950.h5')

inputfile = "input/carname.txt"

# Create header for saving DQN learning file
now = datetime.datetime.now()  # Getting the latest datetime
header = ["Step", "Reward", "Cover_area", "Sharing_factor", "Sent_pkg"]  # Defining header for the save file
filename = "Data/data_test_" + now.strftime("%Y%m%d-%H%M") + ".csv"
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(
        f, encoding='utf-8', index=False, header=True)

total_node, nodes, min_x, max_x, min_y, max_y = read_data(inputfile)
print(total_node, min_x, max_x, min_y, max_y)
gnb = GnbServer(total_node=total_node)
net = Network(list_node=nodes, num_node=total_node, gnb=gnb,
              step_length=param.cover_time, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)

state = net.get_state()
for step in range(MAX_STEP):
    state = tf.convert_to_tensor(state, dtype=tf.float32)
    print(f'state: {state}')
    actions_probs, critic_value = acAgent.forward(state)

    actions = acAgent.act(actions_probs)
    print(f'action: {actions}')

    reward, cover_area, sharing_factor, sent_factor = net.step(actions, step, 1, test=True)
    print(f'reward: {reward}')

    avg_reward = np.average(reward)
    next_state = net.get_state()
    terminate = net.check_terminate(step)

    state = next_state
    reset_tracking(net, step)

    save_data = np.hstack(
        [step + 1, avg_reward, cover_area, sharing_factor, sent_factor]).reshape(1, 5)

    with open(filename, 'a') as f:
        pd.DataFrame(save_data).to_csv(
            f, encoding='utf-8', index=False, header=False)

    if terminate:
        break
