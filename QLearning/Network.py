from getdata import read_data
from GnbServer import GnbServer
import csv
from scipy.spatial import distance
from tensorflow.python.ops.gen_random_ops import TruncatedNormal
import Parameter as para
import os.path
import numpy as np

from Network_Method import *


class Network:
    def __init__(self, list_node=None, num_node=None, nodes=None, gnb=None, location_file=None, step_length=None, min_x=None, max_x=None, min_y=None, max_y=None):
        self.list_node = list_node
        self.num_node = num_node
        self.nodes = nodes
        self.gnb = gnb
        self.location_file = location_file
        self.step_length = step_length
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.map_grid = get_grid_boundary(self.max_x, self.max_y, self.min_x, self.min_y)
        self.not_tracking = np.zeros((para.n_size * para.n_size, 1))

    def get_prob(self):
        prob_list = [0.0 for i in range(0, self.num_node)]
        for i, node in enumerate(self.list_node):
            prob_list[i] = node.get_prob()
        return prob_list

    def update_prob(self, new_prob=None):
        if new_prob:
            for i, node in enumerate(self.list_node):
                node.update_prob(new_prob[i])
    # def update_node_location(self, new_location = None):
    #     if new_location:
    #         for i, node in enumerate(self.list_node):
    #             node.update_node_location(new_location[i])

    def update_node_position(self, t):
        for node in self.list_node:
            node.update_position(t)

    def communicate(self, com_func=communicate_func):
        return com_func(self)

    def run_per_second(self, optimizer=None, com_func=communicate_func, log_file="./log/logfile.txt"):
        return self.communicate(com_func)

    def simulate(self, start_t=0, optimizer="dqn", com_func=None, maxtime=36000, test=False, logfile="./log/logfile.txt"):
        t = start_t

        while(t < maxtime + start_t):
            self.update_node_position(t)
            is_sent = self.run_per_second(com_func=com_func, log_file=logfile)

            # print(t)
            # nb_package = self.gnb.total_receiving
            if t % self.step_length == 0 and test:
                reward = get_reward_v2(self, self.step_length, is_sent, t)
                cover_area = calculate_area_v3(self, is_sent)

                if self.gnb.total_receiving != 0:
                    uniform_sent_ratio = tf.convert_to_tensor(
                        [1 / self.num_node for i in range(self.num_node)], dtype=tf.float32)
                    real_sent_ratio = tf.convert_to_tensor(
                        [i / self.gnb.total_receiving for i in self.gnb.msg_from_node], dtype=tf.float32)
                    sharing_factor = kl_divergence(uniform_sent_ratio, real_sent_ratio).numpy()
                else:
                    sharing_factor = 0
                sent_factor = self.gnb.total_receiving / (self.step_length * self.num_node)

            t += 1

        if test:
            return reward, cover_area, sharing_factor, sent_factor

        return None

    def reset(self):
        self.not_tracking = np.zeros((para.n_size * para.n_size, 1))
        self.update_node_position(0)

    def get_state(self):
        state = np.zeros((self.num_node, 4))
        for i, node in enumerate(self.list_node):
            state[i] = node.latitude, node.longitude, node.moving_step[0], node.moving_step[1]
            # state[i] = node.latitude, node.longitude
        return state

    def get_reward(self, t):
        return get_reward(self, self.step_length, t=t)

    def check_terminate(self, step):
        if step * self.step_length == para.max_t:
            return True
        return False

    def get_prob(self):
        prob = []
        for node in self.list_node:
            prob.append(node.prob)
        return prob

    def update_nodes_prob(self, new_prob):
        for i in range(len(self.list_node)):
            self.list_node[i].update_prob(new_prob[i])

    def update_node_discrete_prob(self, new_prob):
        for i, node in enumerate(self.list_node):
            node.update_prob(new_prob[i])

    def step(self, action, step, ep, optimizer="dqn", test=False):
        self.update_nodes_prob(action)

        # t = 0
        # while t < self.step_length:
        #     self.run_per_second(t, communicate_func)
        #     t+= 1
        return self.simulate(start_t=step * self.step_length + 1, com_func=communicate_func,
                             maxtime=self.step_length, logfile="./log/logfile_" + str(ep) + ".txt", optimizer=optimizer, test=test)


if __name__ == '__main__':
    inputfile = "input/carname.txt"
    total_node, nodes, min_x, max_x, min_y, max_y = read_data(inputfile)
    gnb = GnbServer(total_node=total_node)
    net = Network(list_node=nodes, num_node=total_node, gnb=gnb,
                  step_length=para.cover_time, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)

    is_sent = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
    # test_kld()
    print(calculate_area_v3(net, is_sent))
