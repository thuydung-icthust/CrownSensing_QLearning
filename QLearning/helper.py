# float section(float h, float r = 1) // returns the positive root of intersection of line y = h with circle centered at the origin and radius r
# {
#     assert(r >= 0); // assume r is positive, leads to some simplifications in the formula below (can factor out r from the square root)
#     return (h < r)? sqrt(r * r - h * h) : 0; // http://www.wolframalpha.com/input/?i=r+*+sin%28acos%28x+%2F+r%29%29+%3D+h
# }
from math import *
import Parameter as para
import numpy as np


def section(h, r=1.0):
    # returns the positive root of intersection of line y = h with circle centered at the origin and radius r
    assert(r >= 0)
    if h < r:
        return sqrt(r * r - h * h)
    else:
        return 0


def g(x, h, r=1.0):
    # indefinite integral of circle segment
    return 0.5 * (sqrt(1 - x * x / (r * r)) * x * r + r * r * asin(x / r) - 2 * h * x)


def area(x0, x1, h, r):
    # area of intersection of an infinitely tall box with left edge at x0, right edge at x1, bottom edge at h and top edge at infinity, with circle centered at the origin with radius r
    if x0 > x1:
        x0, x1 = x1, x0
    s = section(h, r)
    return g(max(-s, min(s, x1)), h, r) - g(max(-s, min(s, x0)), h, r)  # integrate the area


def area_finite(x0, x1, y0, y1, r):
    # area of intersection of an infinitely tall box with left edge at x0, right edge at x1, bottom edge at h and top edge at infinity, with circle centered at the origin with radius r
    if y0 > y1:
        y0, y1 = y1, y0
    if y0 < 0:
        if y1 < 0:
            # the box is completely under, just flip it above and try again
            return area_finite(x0, x1, -y0, -y1, r)
        else:
            # the box is both above and below, divide it to two boxes and go again
            return area_finite(x0, x1, 0, -y0, r) + area_finite(x0, x1, 0, y1, r)
    else:
        assert(y1 >= 0)
        # y0 >= 0, which means that y1 >= 0 also (y1 >= y0) because of the swap at the beginning
        # area of the lower box minus area of the higher box
        return area(x0, x1, y0, r) - area(x0, x1, y1, r)


def get_area(x0, x1, y0, y1, cx, cy, r):
    # area of the intersection of a general box with a general circle
    x0 -= cx
    x1 -= cx
    y0 -= cy
    y1 -= cy
    # get rid of the circle center
    return area_finite(x0, x1, y0, y1, r)


def ratio_intersection(x0, x1, y0, y1, cx, cy, r):
    w_rect = abs(x1 - x0)
    h_rect = abs(y1 - y0)
    w_area = w_rect * h_rect
    return get_area(x0, x1, y0, y1, cx, cy, r) / w_area


def get_grid_boundary(x_max, y_max, x_min, y_min):
    n_size = para.n_size
    total_cell = n_size * n_size
    unit_x = (x_max - x_min) / n_size
    unit_y = (y_max - y_min) / n_size
    cell_boundaries = np.zeros((total_cell, 4))
    for i in range(0, total_cell):
        col = i % n_size
        row = i / n_size

        x0 = x_min + col * unit_x
        y0 = y_min + row * unit_y
        x1 = x0 + unit_x
        y1 = y0 + unit_y

        cell_boundaries[i][:] = x0, x1, y0, y1
    return cell_boundaries

def calculate_area_v3(net, is_sent, tries=para.mc_approximation):
    # monte carlos approximation for area
    m = 0

    idxs = np.argwhere(is_sent)
    for i in range(tries):
        x_rand = np.random.uniform(net.min_x, net.max_x)
        y_rand = np.random.uniform(net.min_y, net.max_y)
        for id in idxs:
            dist = (x_rand - net.list_node[id[0]].latitude)**2 + (y_rand - net.list_node[id[0]].longitude)**2
            if (dist < net.radius**2):
                m += 1
                break
    
    # list_area.append(m / tries)
    return m / tries

def calculate_area_v4(max_x, min_x, max_y, min_y, list_node, radius, is_sent, list_area, tries=para.mc_approximation):
    # monte carlos approximation for area
    m = 0

    idxs = np.argwhere(is_sent)
    for i in range(tries):
        x_rand = np.random.uniform(min_x, max_x)
        y_rand = np.random.uniform(min_y, max_y)
        for id in idxs:
            dist = (x_rand - list_node[id[0]].latitude)**2 + (y_rand - list_node[id[0]].longitude)**2
            if (dist < radius**2):
                m += 1
                break
    
    list_area.append(m / tries)
    return m / tries

def calculate_area_v5(max_x, min_x, max_y, min_y, is_sent,list_node,radius, frequency):
    step_x = (max_x - min_x) / frequency
    step_y = (max_y - min_y) / frequency

    idxs = np.argwhere(is_sent)
    node_cord = []
    for i in idxs:
        node_cord.append([list_node[i[0]].latitude, list_node[i[0]].longitude])

    if len(node_cord) ==0:
        return 0
    m = 0
    x = min_x
    y = min_y

    for i in range(frequency):
        # x = min_x + i * step_x
        for j in range(frequency):
            # y = min_y + j * step_y
            for node in node_cord:
                # print(f'{x} {y}')
                # dist = distance.euclidean((x,y), (list_node[id[0]].latitude, list_node[id[0]].longitude))
                dist = (x- node[0])**2 + (y - node[1])**2
                if (dist < radius):
                    m += 1
                    break
            y += step_y
        x += step_x
        y = min_y
    # list_area.append(m / (frequency * frequency))
    return m / (frequency * frequency)

def calculate_area_v6(is_sent,list_node,radius, frequency):
    node_cord = []
    for i in range(len(list_node)):
        node_cord.append([list_node[i].latitude, list_node[i].longitude])

    max_x = -1e9
    max_y = -1e9
    min_x = 1e9
    min_y = 1e9
    for x, y in node_cord:
        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y

    step_x = (max_x - min_x) / frequency
    step_y = (max_y - min_y) / frequency

    if len(node_cord) ==0:
        return 0
    n = 0 # for total possible area
    m = 0 # for actual area
    p = 0 # for overlap area
    x = min_x
    y = min_y
    is_add = False
    is_ovl = False
    for i in range(frequency):
        # x = min_x + i * step_x
        for j in range(frequency):
            # y = min_y + j * step_y
            is_add = False
            is_ovl = False
            for k, node in enumerate(node_cord):
                # print(f'{x} {y}')
                # dist = distance.euclidean((x,y), (list_node[id[0]].latitude, list_node[id[0]].longitude))
                dist = (x- node[0])**2 + (y - node[1])**2
                if (dist < radius):
                    if not is_add:
                        n += 1
                        is_add = True
                    if is_sent[k] == 1:
                        if not is_ovl:
                            m += 1
                            is_ovl = True
                        else: 
                            p += 1
                            break
            y += step_y
        x += step_x
        y = min_y
    # list_area.append(m / (frequency * frequency))
    return m / n, p / n
if __name__ == '__main__':
    print(get_area(1, 2, 1, 2, 3, 3, 2))
