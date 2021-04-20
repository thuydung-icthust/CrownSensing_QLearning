import math
from scipy.spatial import distance
import Parameter as para
import numpy as np
from GnbServer_Method import get_random_prob, get_uniform_prob, update_node_location

class GnbServer:
    def __init__(self, total_node = None):
        self.total_node = total_node
        self.node_prob = [0.0 for i in range(0,self.total_node)]
        self.total_receiving = 0 #total message received from nodes
        self.msg_from_node = [0 for i in range(0, self.total_node)] #total msg reveived from each node in the network
        self.check_point = [{"Total received": self.total_receiving}]
        self.node_location = [(0,0) for i in range(0, self.total_node)] #location of nodes in the network.


    def receive(self, from_id = None, package=None):
        if package:
            self.total_receiving += 1
            self.msg_from_node[from_id] += 1
    
    def update_prob(self, new_prob = None):
        for i, prob in enumerate(new_prob):
            self.node_prob[i]  = prob

    def reset_prob(self, func=get_random_prob):
        new_prob = func(self)
        self.update_prob(new_prob) 
        return new_prob

    def update_node_location(self, func=update_node_location):
        new_location = func(self)
        self.node_location = new_location


    
    


        