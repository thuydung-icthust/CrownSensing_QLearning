import random
from Package import Package
import Parameter as para
import numpy as np
from helper import *
def uniform_com_func(net):
    for node in net.node:
        if random.random() <= node.prob and node.is_active:
            package = Package()
            node.send(net, package)
            # print(package.path)
    return True


def to_string(net):
    min_energy = 10 ** 10
    min_node = -1
    for node in net.node:
        if node.energy < min_energy:
            min_energy = node.energy
            min_node = node
    min_node.print_node()

def print_node_position(net):
    for node in net.list_node:
        node.description()
def count_package_function(net):
    count = 0
    for node in net.node:
        package = Package(is_energy_info=True)
        node.send(net, package)
        if package.path[-1] == -1:
            count += 1
    return count


def communicate_func(net): 
    # func that the nodes send package to the gnb server
    for node in net.list_node:
        if random.random() <= node.prob and node.is_active:
            package = Package()
            node.send(net = net, package = package)
            net.gnb.receive(from_id = node.id, package = package)
    return True

def get_new_prob(net):
    new_prob = net.gnb.reset_prob()
    net.gnb.update_prob(new_prob)
    return new_prob


def get_current_state(net):
    position_list = []
    for node in net.list_node:
        position_list.append([node.longitude, node.latitude])
    np_position_list = np.asarray(position_list)
    # print(np_position_list)
    return np_position_list

def get_current_map_state(net):
    total_cell = para.n_size * para.n_size
    # reward = [[] for i in range(0, total_cell)]
    state_rw = [0.0 for i in range(0, total_cell)]
    cell_boundaries = get_grid_boundary(net.max_x, net.max_y, net.min_x, net.min_y)
    # print(cell_boundaries)
    # print(cell_boundaries.__len__())
    for i in range(0, total_cell):
        bound = cell_boundaries[i]
        # print(bound.__len__())
        x0, x1, y0, y1 = bound[0], bound[1], bound[2], bound[3]
        for node in net.list_node:
            
            inter_area = ratio_intersection(x0, x1, y0, y1, node.longitude, node.latitude, para.cover_radius)
            if state_rw[i] < inter_area * node.prob:
                state_rw[i] = inter_area * node.prob
    for i in range(0, total_cell):
        if state_rw[i] == 0:
            net.not_tracking[i] += 1
    return state_rw

def get_reward(net, delta_t):
    total_cell = para.n_size * para.n_size
    state_rw = get_current_map_state(net)
    untracking = np.sum(net.not_tracking)
    factor1 = sum(state_rw)/total_cell #covering factor
    factor2 = untracking/(delta_t*total_cell) #total cell untrack
    factor3 = net.gnb.total_receiving/(net.step_length*net.num_node)

    if delta_t == 0:
        factor2 = 0.0

    
    return (para.theta * factor1 - para.gamma * factor2) + 0.1*factor3


def reset_tracking(net):
    net.not_tracking = np.zeros((para.n_size*para.n_size, 1))
    net.gnb.msg_from_node = [0 for i in range(0, net.gnb.total_node)]
    net.gnb.total_receiving = 0



