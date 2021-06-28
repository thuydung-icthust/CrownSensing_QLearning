import config as dqn_conf
from tensorflow.keras.models import model_from_json
from DQNModel import DQN
from Node import Node
from Network import Network
from GnbServer import GnbServer
from getdata import read_data
from Network_Method import get_current_map_state, get_reward, uniform_com_func, communicate_func, print_node_position, get_current_state, get_current_map_state, get_reward, reset_tracking
import datetime
import numpy as np
import pandas as pd


maxtime = 3600
input_num = 16

# load model from saved file
json_file = open('./TrainedModels/DQNmodel_20210514-1452_ep200.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('./TrainedModels/DQNmodel_20210514-1452_ep200.h5')

DQNAgent = DQN(input_num, dqn_conf.ACTIONNUM, model=model)

print('Model loaded from disk')


inputfile = "input/carname.txt"
# Create header for saving DQN learning file
now = datetime.datetime.now()  # Getting the latest datetime
header = ["Ep", "Step", "Reward", "Total_reward", "Action", "Epsilon",
          "Done", "Termination_Code"]  # Defining header for the save file
filename = "./TestData/data_" + now.strftime("%Y%m%d-%H%M") + ".csv"
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(
        f, encoding='utf-8', index=False, header=True)

total_node, nodes, min_x, max_x, min_y, max_y = read_data(inputfile)
print(total_node, min_x, max_x, min_y, max_y)
gnb = GnbServer(total_node=total_node)
net = Network(list_node=nodes, num_node=total_node, gnb=gnb,
              step_length=30, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
prob_list = net.get_prob()
net.update_prob(prob_list)


# start running the testing process
try:
    net.reset()
    s = net.get_state()
    total_reward = 0  # The amount of rewards for the entire episode
    step = 0
    while not net.check_terminate(step):
        try:
            action = DQNAgent.act(s)
            print("next action = ", action)
            # Performing the action in order to obtain the new state
            net.step(action, step, 1, "dqn")
            s_next = net.get_state()  # Getting a new state

            reward = net.get_reward(
                t=step * net.step_length)  # Getting a reward
            terminate = net.check_terminate(
                step)  # checking the end status
            total_reward = total_reward + reward

            s = s_next
            save_data = np.hstack(
                [1, step + 1, reward, total_reward, action, DQNAgent.epsilon, terminate]).reshape(1, 7)
            with open(filename, 'a') as f:
                pd.DataFrame(save_data).to_csv(
                    f, encoding='utf-8', index=False, header=False)

            step += 1

        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Finished.")
            break

except Exception as e:
    import traceback
    traceback.print_exc()
print("End game.")
