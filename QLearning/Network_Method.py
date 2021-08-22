import math
import random

import tensorflow as tf
from Package import Package
import Parameter as para
import numpy as np
from helper import *
from tensorflow.keras.metrics import kl_divergence


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
    is_sent = np.zeros(net.num_node)
    for node in net.list_node:
        if random.random() <= node.prob and node.is_active:
            package = Package()
            node.send(net=net, package=package)
            net.gnb.receive(from_id=node.id, package=package)
            is_sent[node.id] = 1
    return is_sent


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
    cell_boundaries = get_grid_boundary(
        net.max_x, net.max_y, net.min_x, net.min_y)
    # print(cell_boundaries)
    # print(cell_boundaries.__len__())
    for i in range(0, total_cell):
        bound = cell_boundaries[i]
        # print(bound.__len__())
        x0, x1, y0, y1 = bound[0], bound[1], bound[2], bound[3]
        for node in net.list_node:

            inter_area = ratio_intersection(
                x0, x1, y0, y1, node.longitude, node.latitude, net.radius)
            if state_rw[i] < inter_area * node.prob:
                state_rw[i] = inter_area * node.prob
    for i in range(0, total_cell):
        if state_rw[i] == 0:
            net.not_tracking[i] += 1
    return state_rw


def get_reward(net, delta_t, t=0, logfile="log/dqn_logfile.txt"):
    total_cell = para.n_size * para.n_size
    state_rw = get_current_map_state(net)
    untracking = np.sum(net.not_tracking) * net.step_length
    factor1 = sum(state_rw) / total_cell  # covering factor
    factor2 = untracking / (delta_t * total_cell)  # total cell untrack
    factor3 = net.gnb.total_receiving / (net.step_length * net.num_node)
    with open(logfile, "a+") as f:
        str_to_log = str(t) + "," + str(factor1) + "," + str(factor2) + \
            "," + str(net.gnb.total_receiving) + "\n"
        f.write(str_to_log)
        f.close()
    if delta_t == 0:
        factor2 = 0.0

    return (para.theta * factor1 - para.gamma * factor2) - 0.025 * factor3


def calculate_cover_area(net, idx, is_sent):
    factor1 = 0
    if (np.all(np.array(is_sent) == 0)):
        return factor1

    inode = net.list_node[idx]
    area = np.pi * (net.radius**2)
    
    def overlap_area(x1, y1, x2, y2, R=net.radius, r=net.radius):
        d = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if d == 0:
            # One circle is entirely enclosed in the other.
            return np.pi * min(R, r)**2
        if d >= r + R:
            # The circles don't overlap at all.
            return 0

        r2, R2, d2 = r**2, R**2, d**2
        alpha = np.arccos((d2 + r2 - R2) / (2 * d * r))
        beta = np.arccos((d2 + R2 - r2) / (2 * d * R))
        return (r2 * alpha + R2 * beta -
                0.5 * (r2 * np.sin(2 * alpha) + R2 * np.sin(2 * beta))
                )

    if (is_sent[idx] == 1):
        factor1 = 1

    for node in net.list_node:
        # if it is the investigating node or the
        if (idx == node.id or is_sent[node.id] == 0):
            continue
        else:
            ovl_area = overlap_area(inode.latitude, inode.longitude, node.latitude,
                                        node.longitude) / area
            # print(f'overlap area: {ovl_area}, factor 1: {factor1}')
            if (is_sent[idx] == 1):
                factor1 -= ovl_area
            else:
                factor1 += ovl_area

    return factor1

def get_reward_v2(net, acted_agent, factor1, delta_t, logfile="log/dqn_logfile.txt"):
    rewards = [0 for i in range(net.num_node)]

    for idx in acted_agent:
        # factor1 = calculate_cover_area(net, idx, is_sent)  # cover area, take into account the overlapping area
        if net.gnb.total_receiving != 0:
            factor2 = abs((net.gnb.msg_from_node[idx] / net.gnb.total_receiving) -
                          (1 / net.num_node))  # sent ratio / uniform ratio
        else:
            factor2 = 0
        if delta_t == 0:
            factor3 = 0
        else:
            factor3 = net.gnb.msg_from_node[idx] / delta_t

        # if (factor1 > 0.85):  # if the cover factor is small, add additional weight to this factor to push it up
        #     rewards[idx] = para.thetab * factor1 - para.gammab * factor2 - para.sigmab * factor3
        # else:
        rewards[idx] = para.theta * factor1 - para.gamma * factor2 - para.sigma * factor3
        # print(f'factor 1: {factor1} factor 2: {factor2} factor 3: {factor3}')
    return rewards


# def get_reward_v3(net, delta_t, is_sent, t=0, logfile="log/dqn_logfile.txt"):
#     rewards = 0

#     factor1 = calculate_cover_area_v2(net, is_sent)  # cover area, take into account the overlapping area

#     uniform_sent_ratio = tf.convert_to_tensor([1 / net.num_node for i in range(net.num_node)])
#     real_sent_ratio = tf.convert_to_tensor([i / net.gnb.total_receiving for i in net.gnb.msg_from_node])
#     factor2 = kl_divergence(uniform_sent_ratio, real_sent_ratio).numpy()

#     factor3 = net.gnb.total_receiving / (delta_t * net.num_node)

#     rewards = para.theta * factor1 - para.gamma * factor2 - para.sigma * factor3

#     return rewards


def reset_tracking(net):
    net.not_tracking = np.zeros((para.n_size * para.n_size, 1))
    net.gnb.msg_from_node = [0 for i in range(0, net.gnb.total_node)]
    net.gnb.total_receiving = 0


def test_kld():
    a = tf.convert_to_tensor(np.random.rand(10))
    b = tf.convert_to_tensor(np.random.rand(10))
    print(kl_divergence(a, b).numpy())

