import numpy as np
from scipy.spatial import distance
import math
import Parameter as para


def to_string(node):
    print("Id =", node.id, "RefName =", node.car_name, "prob = ", node.prob, "Location long =", node.longitude, "Location lat =", node.latitude, "Energy =", node.energy,
          "Total time sent: ", node.total_sending)


def find_receiver(node):
    return -1


def update_postion(node, t):
    start_t = para.min_time_step
    end_t = para.max_time_step
    # if t == 0:
    #     node.update_loc(long = node.location_list[1][0], lat = node.location_list[2][0])
    # else:
    #     start
    if t % 600 == 0:
        if t == 0:
            node.current_row = 0
        elif (node.current_row < node.location_list.shape[0] - 1) and node.location_list[0][node.current_row + 1] == t + start_t:
            node.current_row += 1

        while (node.current_row < node.location_list.shape[0] - 1) and node.location_list[0][node.current_row + 1] == node.location_list[0][node.current_row]:
            node.current_row += 1

        if (node.current_row < node.location_list.shape[0] - 1):
            current_x = node.location_list[1][node.current_row]
            current_y = node.location_list[2][node.current_row]
            node.update_loc(long=current_x, lat=current_y)
            node.moving_step = update_moving_step(node)
        else:
            current_x = node.location_list[1][node.current_row]
            current_y = node.location_list[2][node.current_row]
            node.update_loc(long=current_x, lat=current_y)
            # if the last row, keep the same direction
    else:
        current_x = node.longitude + node.moving_step[0]
        current_y = node.latitude + node.moving_step[1]
        node.update_loc(long=current_x, lat=current_y)


# def find_receiver(node, net):
#     """
#     find receiver node
#     :param node: node send this package
#     :param net: network
#     :return: find node nearest base from neighbor of the node and return id of it
#     """
#     if not node.is_active:
#         return -1
#     list_d = [distance.euclidean(para.base, net.node[neighbor_id].location) if net.node[
#         neighbor_id].is_active else float("inf") for neighbor_id in node.neighbor]
#     id_min = np.argmin(list_d)
#     if distance.euclidean(node.location, para.base) <= list_d[id_min]:
#         return -1
#     else:
#         return node.neighbor[id_min]

def update_moving_step(node):
    distance_x = node.location_list[1][node.current_row +
                                       1] - node.location_list[1][node.current_row]
    distance_y = node.location_list[2][node.current_row +
                                       1] - node.location_list[2][node.current_row]
    time_moving = node.location_list[0][node.current_row +
                                        1] - node.location_list[0][node.current_row]
    return (distance_x / time_moving, distance_y / time_moving)


def find_receiver(node, net):
    if not node.is_active:
        return -1
    # candidate = [neighbor_id for neighbor_id in node.neighbor if
    #              net.node[neighbor_id].level < node.level and net.node[neighbor_id].is_active]
    # if candidate:
    #     d = [distance.euclidean(net.node[candidate_id].location, para.base) for candidate_id in candidate]
    #     id_min = np.argmin(d)
    #     return candidate[id_min]
    # else:
    #     return -1
    return 0  # 0 means the gnb server


def request_function(node, mc, t):
    mc.list_request.append(
        {"id": node.id, "energy": node.energy, "avg_energy": node.avg_energy, "energy_estimate": node.energy,
         "time": t})


def estimate_average_energy(node):
    return node.check_point[-1]["avg_e"]
