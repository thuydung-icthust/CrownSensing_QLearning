import numpy as np
from scipy.spatial import distance
import Parameter as para
from math import *
from Node_Method import find_receiver
from Q_learning_method import (
    init_function,
    action_function,
    q_max_function,
    reward_function,
    discrete_action_function,
)


class Q_learning:
    def __init__(
        self,
        init_func=init_function,
        nb_action=100,
        action_func=discrete_action_function,
        network=None,
        alpha_p=0.1,
        theta_p=0.3,
        n_size=10,
    ):
        self.nb_action = nb_action
        print(self.nb_action)
        self.action_list = action_func(nb_action=nb_action+1, n_size=n_size)  # the list of action
        self.q_table = init_func(nb_action=nb_action)  # q table
        print(len(self.action_list))
        self.state = nb_action  # the current state of actor
        self.charging_time = [
            0.0 for _ in self.action_list
        ]  # the list of charging time at each action
        self.reward = np.asarray(
            [0.0 for _ in self.action_list]
        )  # the reward of each action
        self.reward_max = [
            0.0 for _ in self.action_list
        ]  # the maximum reward of each action
        self.alpha_p = alpha_p
        self.theta_p = theta_p
        self.n_size = n_size
        
        print(self.alpha_p)
        print(self.theta_p)

    def update(
        self,
        network,
        alpha=0.1,
        gamma=0.5,
        q_max_func=q_max_function,
        reward_func=reward_function,

    ):
        self.update_action_list(network)
        if not len(network.mc.list_request):
            return self.action_list[self.state], 0.0
        self.set_reward(reward_func=reward_func, network=network)
        self.q_table[self.state] = (1 - self.alpha_p) * self.q_table[self.state] + self.alpha_p * (
            self.reward + gamma * self.q_max(q_max_func)
        )
        self.choose_next_state(network)
        if self.state == len(self.action_list) - 1:
            charging_time = (
                network.mc.capacity - network.mc.energy
            ) / network.mc.e_self_charge
        else:
            charging_time = self.charging_time[self.state]

        # print(self.charging_time)
        return self.action_list[self.state], int(charging_time)

    def update_action_list(self, network):
        request_list = []
        n_size = self.n_size
        print(self.n_size)
        total_cell = n_size * n_size
        candidates = [[] for i in range(0, total_cell)]
        for node in network.node:
            if node.is_request:
                request_list.append(node)
        circle_circle_intersection_points = get_circle_circle_intersection(request_list, n_size=self.n_size)
        circle_line_intersection_points = get_circle_line_intersection(request_list, n_size=self.n_size)
        circle_bound_grid_points = get_circle_bound_grid(request_list, n_size=self.n_size)
        for i in range(0, len(candidates)):
            candidates[i].extend(circle_circle_intersection_points[i])
            candidates[i].extend(circle_line_intersection_points[i])
            candidates[i].extend(circle_bound_grid_points[i])
        action_list = optimal_action_list(candidates, network, self.action_list, nb_action=self.nb_action)

        self.action_list = action_list.copy()
        # self.action_list.append(para.depot)
        

    def q_max(self, q_max_func=q_max_function):
        return q_max_func(q_table=self.q_table, state=self.state)

    def set_reward(self, reward_func=reward_function, network=None):
        first = np.asarray([0.0 for _ in self.action_list], dtype=float)
        second = np.asarray([0.0 for _ in self.action_list], dtype=float)
        # third = np.asarray([0.0 for _ in self.action_list], dtype=float)
        for index, row in enumerate(self.q_table):
            temp = reward_func(
                network=network,
                q_learning=self,
                state=index,
                receive_func=find_receiver,
                alpha=self.theta_p,

            )
            first[index] = temp[0]
            second[index] = temp[1]
            # third[index] = temp[2]
            self.charging_time[index] = temp[2]
        first = first / np.sum(first)
        second = second / np.sum(second)
        # third = third / np.sum(third)
        self.reward = first + second
        self.reward_max = list(zip(first, second))

    def choose_next_state(self, network):
        # next_state = np.argmax(self.q_table[self.state])
        if network.mc.energy < 10:
            self.state = len(self.q_table) - 1
        else:
            self.state = np.argmax(self.q_table[self.state])
            print(self.reward_max[self.state])
            print(self.action_list[self.state])


def get_circle_bound_grid(request_list, n_size):
    # n_size = self.n_size
    total_cell = n_size * n_size
    candidates = [[] for i in range(0, total_cell)]
    unit_x = (para.x_bound[1] - para.x_bound[0]) / n_size
    unit_y = (para.y_bound[1] - para.y_bound[0]) / n_size
    for i in range(0, n_size):
        for j in range(0, n_size):
            for node in request_list:
                x, y, r = (
                    node.location[0],
                    node.location[1],
                    get_positive_charging_radius(node),
                )
                if(isInside(x, y, r, i*n_size, j*n_size)):
                    candidates[i*n_size+j].append([i*n_size, j*n_size])
                if(isInside(x, y, r, (i+1)*n_size, j*n_size)):
                    candidates[i*n_size+j].append([(i+1)*n_size, j*n_size])
    return candidates


def get_circle_circle_intersection(request_list, n_size):
    n_size = n_size
    total_cell = n_size * n_size
    candidates = [[] for i in range(0, total_cell)]
    for i in range(0, len(request_list) - 1):
        for j in range(i + 1, len(request_list)):
            node1 = request_list[i]
            node2 = request_list[j]
            intersections = get_circle_intersections(
                node1.location[0],
                node1.location[1],
                get_positive_charging_radius(node1),
                node2.location[0],
                node2.location[1],
                get_positive_charging_radius(node2),
            )
            if intersections is not None:
                x1, y1, x2, y2 = intersections
                cell_num1 = isBelongToCell(x1, y1, n_size=n_size)
                cell_num2 = isBelongToCell(x2, y2, n_size=n_size)
                if cell_num1 != -1:
                    candidates[cell_num1].append([x1, y1])
                if cell_num2 != -1:
                    candidates[cell_num2].append([x2, y2])
    return candidates


def get_circle_line_intersection(request_list, n_size):
    # n_size = n_size
    total_cell = n_size * n_size
    candidates = [[] for i in range(0, total_cell)]
    unit_x = (para.x_bound[1] - para.x_bound[0]) / n_size
    unit_y = (para.y_bound[1] - para.y_bound[0]) / n_size
    for i in range(0, n_size):
        # the line y = c
        y = i * unit_y
        for node in request_list:
            intersections = get_line_intersections(
                node.location[0],
                node.location[1],
                get_positive_charging_radius(node),
                0,
                1,
                -y,
            )
            if len(intersections) >= 1:
                cell_num = isBelongToCell(intersections[0][0], intersections[0][1], n_size=n_size)
                if cell_num != -1:
                    candidates[cell_num].append(intersections[0])
            if len(intersections) >= 2:
                cell_num = isBelongToCell(intersections[1][0], intersections[1][1], n_size=n_size)
                if cell_num != -1:
                    candidates[cell_num].append(intersections[1])
    for i in range(0, n_size):
        # the line x = c
        x = i * unit_x
        for node in request_list:
            intersections = get_line_intersections(
                node.location[0],
                node.location[1],
                get_positive_charging_radius(node),
                1,
                0,
                -x,
            )
            if len(intersections) >= 1:
                cell_num = isBelongToCell(intersections[0][0], intersections[0][1], n_size=n_size)
                if cell_num != -1:
                    candidates[cell_num].append(intersections[0])
            if len(intersections) >= 2:
                cell_num = isBelongToCell(intersections[1][0], intersections[1][1], n_size=n_size)
                if cell_num != -1:
                    candidates[cell_num].append(intersections[1])

    return candidates


def get_circle_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # non intersecting
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0 - r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = sqrt(r0 ** 2 - a ** 2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        return (x3, y3, x4, y4)


def solveQuad(a, b, c, a0, b0, c0):
    # print(a,b,c)
    result = []
    d = b ** 2 - 4 * a * c
    if(d < 0):
        return result
    d1 = sqrt(d)
    # print(d1)
    if d < 0:
        return
    elif d > 0:
        y1 = (-b + d1) / (2 * a)
        y2 = (-b - d1) / (2 * a)
        # print(y1)
        if a0 == 0:
            x1 = -c0 / b0
            # result = [[y1,x1], [y2,x1]]
            result.extend([[y1, x1], [y2, x1]])
        else:
            x1 = (-b0 * y1 - c0) / a0

            x2 = (-b0 * y2 - c0) / a0
            # result = [[x1,y1],[x2,y2]]
            result.extend([[x1, y1], [x2, y2]])
    else:
        y1 = (-b) / 2 * a
        if a0 == 0:
            x1 = -c0 / b0
            result.append([y1, x1])
        else:
            x1 = (-b0 * y1 - c0) / a0
            result.append([x1, y1])
    return result


def get_line_intersections(x, y, r, a0, b0, c0):

    if a0 == 0:
        result = solveQuad(
            1, -2 * x, x ** 2 + ((c0 / b0) + y) ** 2 - r ** 2, a0, b0, c0
        )
    else:
        result = solveQuad(
            (b0 / a0) ** 2 + 1,
            2 * ((b0 / a0) * ((c0 / a0) + x) - y),
            ((c0 / a0) + x) ** 2 + y ** 2 - r ** 2,
            a0,
            b0,
            c0,
        )
    # print(result)
    return result


def get_positive_charging_radius(node):
    e = node.avg_energy
    if e == 0 :
        return 100
    return max(0, sqrt(para.alpha / e) - para.beta)


def isBelongToCell(x, y, n_size):
    if (
        x >= para.x_bound[0]
        and x <= para.x_bound[1]
        and y >= para.y_bound[0]
        and y <= para.y_bound[1]
    ):
        unit_x = (para.x_bound[1] - para.x_bound[0]) / n_size
        unit_y = (para.y_bound[1] - para.y_bound[0]) / n_size
        i = int(x / unit_x)
        j = int(y / unit_y)
        return i * n_size + j
    return -1

def isInside(circle_x, circle_y, rad, x, y): 
      
    # Compare radius of circle 
    # with distance of its center 
    # from given point 
    if ((x - circle_x) * (x - circle_x) + 
        (y - circle_y) * (y - circle_y) <= rad * rad): 
        return True
    else: 
        return False


def optimal_action_list(candidates, network, initial_action_list, nb_action):
    node_positions = [[node.location[0], node.location[1]] for node in network.node]
    # node_positions = np.asarray(node_positions)
    action_list = [[0,0] for i in range (0, len(candidates)+1)]
    e = [node.avg_energy for node in network.node]
    e = np.asarray(e)
    for ind, actions in enumerate(candidates):
        if(len(actions) == 0):
            action_list[ind] = initial_action_list[ind]
        else:
            evaluations = [[0.0,0.0,0.0,0.0] for i in range(0, len(actions))]
            evaluations = np.asarray(evaluations)
            for j, action in enumerate(actions):
                dist = [0 for i in range (0, len(node_positions))]
                # for i, pos in enumerate(node_positions):
                #     dist[i] = distance(pos, action)
                dist = [distance.euclidean(pos, action) for pos in node_positions]
                dist = np.asarray(dist)
                N0, total_p = estimate_charging(dist, network, e)
                evaluations[j][0] = N0
                evaluations[j][1] = total_p
                evaluations[j][2] = action[0]
                evaluations[j][3] = action[1]
            minus_eval = - evaluations
            if len(evaluations) > 1:
                evaluations = evaluations[np.argsort(minus_eval[:,1])] # sort by decreasing total p
            action_list[ind] = (int(evaluations[np.argmax(evaluations[:,0])][2]), int(evaluations[np.argmax(evaluations[:,0])][3]))
            action_list[nb_action] = initial_action_list[nb_action]
    return action_list


def estimate_charging(dist, network, e):
    p = para.alpha/((dist+para.beta)**2)
    total_p = sum(p)

    N0 = np.count_nonzero(p>e)
    return N0, total_p