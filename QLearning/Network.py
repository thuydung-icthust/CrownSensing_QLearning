import csv
from scipy.spatial import distance
from tensorflow.python.ops.gen_random_ops import TruncatedNormal
import Parameter as para
import os.path
import numpy as np

from Network_Method import get_current_map_state, get_reward, uniform_com_func, communicate_func, print_node_position, get_current_state, get_current_map_state, get_reward, reset_tracking

class Network:
    def __init__(self, list_node=None, num_node = None, nodes = None, gnb = None, location_file = None, step_length = None, min_x = None, max_x = None, min_y = None, max_y = None):
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
        self.not_tracking = np.zeros((para.n_size*para.n_size, 1))
        
    def get_prob(self):
        prob_list = [0.0 for i in range(0, self.num_node)]
        for i, node in enumerate(self.list_node):
            prob_list[i] = node.get_prob()
        return prob_list
    
    def update_prob(self, new_prob = None):
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

    def communicate(self, com_func = communicate_func):
        com_func(self)
        return True

    def run_per_second(self, t, optimizer = None, com_func = communicate_func, log_file = "./log/logfile.txt"):
        self.communicate(com_func)
        
    def simulate(self,start_t = 0, optimizer = None, com_func = None, maxtime = 36000, logfile = "./log/logfile.txt"):
        t = start_t
        while(t < maxtime+start_t):
            self.update_node_position(t)
            nb_package = self.gnb.total_receiving
            # if os.path.isfile(logfile):
            #     f = open(logfile, "a+")
            # else:
            # str_to_log = str("t = ", t, "Total package receive: \n", nb_package)
            if t%self.step_length == 0:
                str_to_log = "\nt = " + str(t) + " Total package receive: " + str(nb_package)
                with open(logfile, "a+") as f:
                    f.write(str_to_log)
                    f.close()
            # if t % 100 == 0:
            #    print(str_to_log)
            if t % self.step_length == 0:
                get_current_state(self)
                # print(get_current_map_state(self))
                # print(get_reward(self, self.step_length))
                # reset_tracking(self)
            #if t % 100 == 0:
                #print_node_position(self)
            
            self.run_per_second(t, optimizer, com_func, logfile)
            t+=1

    def reset(self):
        self.not_tracking = np.zeros((para.n_size*para.n_size, 1))
        self.update_node_position(0)
    def get_state(self):
        state = np.zeros((self.num_node, 2))
        for i, node in enumerate(self.list_node):
            state[i] = node.longitude, node.latitude
        state = np.reshape(state, (self.num_node*2,1))
        return state
    
    def get_reward(self):
        return get_reward(self, self.step_length)
    
    def check_terminate(self, step):
        if step*self.step_length == para.max_t:
            return True
        return False

    def step(self, action, step, ep):
        new_prob = 0.1*(action+1)
        for node in self.list_node:
            node.update_prob(new_prob)
        # t = 0
        # while t < self.step_length:
        #     self.run_per_second(t, communicate_func)
        #     t+= 1
        self.simulate(start_t=step*self.step_length+1, com_func=communicate_func, maxtime=self.step_length, logfile="./log/logfile_" + str(ep) + ".txt")

    

    




