from Node import Node
from Network import Network
import pandas as pd
from ast import literal_eval
from GnbServer import GnbServer
from getdata import read_data
from Network_Method import communicate_func

inputfile = "input/carname.txt"

total_node, nodes, min_x, max_x, min_y, max_y = read_data(inputfile)
print (total_node, min_x, max_x, min_y, max_y)
gnb = GnbServer(total_node = total_node)
net = Network(list_node = nodes, num_node = total_node, gnb = gnb, step_length = 30, min_x = min_x, max_x = max_x, min_y = min_y, max_y = max_y)
prob_list = net.get_prob()
net.update_prob(prob_list)
net.simulate(com_func=communicate_func, maxtime=36000)
