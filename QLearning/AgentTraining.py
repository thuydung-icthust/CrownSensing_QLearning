import config as dqn_conf
from Memory import Memory
from DQNModel import DQN
from Node import Node
from Network import Network
import pandas as pd
from ast import literal_eval
from GnbServer import GnbServer
from getdata import read_data
from Network_Method import get_current_map_state, get_reward, uniform_com_func, communicate_func, print_node_position, get_current_state, get_current_map_state, get_reward, reset_tracking
import datetime
import numpy as np

maxtime = 3600
num_nodes = 8
dqnAgent = DQN(num_nodes, dqn_conf.ACTIONNUM)
memory = Memory(dqn_conf.MEMORY_SIZE)

inputfile = "input/carname.txt"
# Create header for saving DQN learning file
now = datetime.datetime.now()  # Getting the latest datetime
header = ["Ep", "Step", "Reward", "Total_reward", "Action_1", "Action_2", "Epsilon",
          "Done", "Termination_Code"]  # Defining header for the save file
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv"
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(
        f, encoding='utf-8', index=False, header=True)

total_node, nodes, min_x, max_x, min_y, max_y = read_data(inputfile)
print(total_node, min_x, max_x, min_y, max_y)
gnb = GnbServer(total_node=total_node)
net = Network(list_node=nodes, num_node=total_node, gnb=gnb,
              step_length=30, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
prob_list = net.get_prob()
# net.update_prob(prob_list)
# net.simulate(com_func=communicate_func, maxtime=36000)

train = False
# The variable is used to indicate that the replay starts, and the epsilon starts decrease.
# Training Process
# the main part of the deep-q learning agorithm
for episode_i in range(0, dqn_conf.N_EPISODE):
    try:
        # Getting the initial state
        # net.update_node_position(0)
        net.reset()
        s = net.get_state()

        total_reward = 0  # The amount of rewards for the entire episode
        terminate = False
        maxStep = dqn_conf.MAX_STEP
        # Start an episde for training

        step = 0  # tracking time
        while step <= maxStep:
            # get_current_state(net)
            # Getting an action from the DQN model from the state (s)
            action = dqnAgent.act(s)
            print(f'action: {action}')

            # Performing the action in order to obtain the new state
            reward = net.step(action, step, episode_i, "dqn")
            print(f'reward: {reward}')

            s_next = net.get_state()  # Getting a new state
            terminate = net.check_terminate(
                step)  # checking the end status

            reset_tracking(net, step)
            # Add this transition to the memory batch
            # print(f'state2: {s}')
            memory.push(s, action, reward, terminate, s_next)
            if (memory.length > dqn_conf.INITIAL_REPLAY_SIZE):
                # Get a BATCH_SIZE experiences for replaying
                batch = memory.sample(dqn_conf.BATCH_SIZE)
                dqnAgent.replay(batch, dqn_conf.BATCH_SIZE)  # Do relaying
                train = True
            # Plus the reward to the total rewad of the episode
            total_reward = total_reward + reward
            s = s_next  # Assign the next state for the next step.
            save_data = np.hstack(
                [episode_i + 1, step + 1, np.average(reward), np.average(total_reward), action[0],
                 action[1], dqnAgent.epsilon, terminate]).reshape(1, 8)
            with open(filename, 'a') as f:
                pd.DataFrame(save_data).to_csv(
                    f, encoding='utf-8', index=False, header=False)

            if terminate == True:
                # If the episode ends, then go to the next episode
                break

            # net.run_per_second(t, com_func = communicate_func)
            step += 1

        # Iteration to save the network architecture and weights
        if (np.mod(episode_i + 1, dqn_conf.SAVE_NETWORK) == 0 and train == True):
            dqnAgent.target_train()
            # Save the DQN model
            now = datetime.datetime.now()  # Get the latest datetime
            dqnAgent.save_model("TrainedModels/",
                                "DQNmodel_" + now.strftime("%Y%m%d-%H%M") + "_ep" + str(episode_i + 1))

        # Print the training information after the episode
        print('Episode %d ends. Number of steps is: %d. Accumulated Reward = %.2f. Epsilon = %.2f .Termination code: %d' % (
            episode_i + 1, step + 1, total_reward, dqnAgent.epsilon, terminate))

        # Decreasing the epsilon if the replay starts
        if train == True:
            dqnAgent.update_epsilon()

    except Exception as e:
        import traceback

        traceback.print_exc()
        # print("Finished.")
        break
