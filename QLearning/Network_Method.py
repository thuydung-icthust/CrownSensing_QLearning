import math
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
                x0, x1, y0, y1, node.longitude, node.latitude, para.cover_radius)
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
    inode = net.list_node[idx]

    def circle_area(r=para.cover_radius):
        return np.pi * r * r

    def overlap_area(x1, y1, x2, y2, R=para.cover_radius, r=para.cover_radius):
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
            if (is_sent[idx] == 1):
                factor1 -= overlap_area(inode.latitude, inode.longitude, node.latitude,
                                        node.longitude) / circle_area()
            else:
                factor1 += overlap_area(inode.latitude, inode.longitude, node.latitude,
                                        node.longitude) / circle_area()

    return factor1


def get_reward_v2(net, delta_t, is_sent, t=0, logfile="log/dqn_logfile.txt"):
    rewards = np.zeros(net.num_node)

    for idx in range(net.num_node):
        factor1 = calculate_cover_area(net, idx, is_sent)  # cover area, take into account the overlapping area
        factor2 = abs((net.gnb.msg_from_node[idx] / net.gnb.total_receiving) -
                      (net.gnb.total_receiving / net.num_node))  # sent ratio / uniform ratio
        factor3 = is_sent[idx]
        rewards[idx] = para.theta * factor1 - para.gamma * factor2 - para.sigma * factor3

    return rewards


def reset_tracking(net, step):
    net.not_tracking = np.zeros((para.n_size * para.n_size, 1))
    # only reset the # of msg after each 30min
    if step % 60 == 0:
        net.gnb.msg_from_node = [0 for i in range(0, net.gnb.total_node)]
        net.gnb.total_receiving = 0
