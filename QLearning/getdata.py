import csv
import pandas as pd
import numpy as np
from Node import Node
import Parameter as para


def getCarData(filename="./input/carname.txt"):
    pf = open(filename, "r+")
    if pf == None:
        return False
    lines = pf.readlines()
    for line in lines:
        car_name = line.strip()
        print(car_name)
        input_car_file = "./input/MTA_Tracking/" + car_name
        car_data = pd.read_csv(filepath_or_buffer=input_car_file, error_bad_lines=False)
        df = pd.DataFrame(car_data)


def read_data(filename="./input/carname.txt"):
    nodes = []
    max_x = -1e10
    min_x = 1e10
    max_y = -1e10
    min_y = 1e10
    pf = open(filename, "r+")

    if pf == None:
        return False
    lines = pf.readlines()
    node_pos = []
    for line in lines:
        car_name = line.strip()
        # print(car_name)
        input_car_file = "./input/MTA_Tracking/" + car_name
        t_min_x, t_max_x, t_min_y, t_max_y, name, positions = read_csv_data(input_car_file)
        node_pos.append(positions)

        if t_min_x < min_x:
            min_x = t_min_x
        if t_min_y < min_y:
            min_y = t_min_y
        if t_max_x > max_x:
            max_x = t_max_x
        if t_max_y > max_y:
            max_y = t_max_y

    for index in range(len(node_pos)):
        positions = node_pos[index]
        # print(positions.shape)
        positions[1,:] = positions[1,:] / max(max_x, abs(min_x))
        positions[2,:] = positions[2,:] / max(max_y, abs(min_y))
        node = Node(car_name=name, location_list=node_pos[index], id=index)
        node.description()
        nodes.append(node)

    norm_radius = para.cover_radius / max(max_x, max_y)
    max_x = max_x / max(max_x, abs(min_x))
    min_x = min_x / max(max_x, abs(min_x))
    min_y = min_y / max(max_y, abs(min_y))
    max_y = max_y / max(max_y, abs(min_y))
    
    print(max_x, min_x, min_y, max_y, norm_radius)
    return len(node_pos), nodes, min_x, max_x, min_y, max_y, norm_radius


def read_csv_data(filename):
    max_x = -1e10
    min_x = 1e10
    max_y = -1e10
    min_y = 1e10
    car_data = pd.read_csv(filepath_or_buffer=filename, error_bad_lines=False)
    car_data.head()
    car_name = car_data['VehicleRef'].values[0]
    time_recored = car_data['RecordedAtTime']
    long, lat = car_data['VehicleLocation.Longitude'].values, car_data['VehicleLocation.Latitude'].values

    # long_norm = long / abs(long).max()
    # lat_norm = lat / abs(lat).max()
    # print(long_norm)
    # print(lat_norm)
    # return_value = np.vstack((time_recored.values, lat_norm, long_norm))
    return_value = np.vstack((time_recored.values, lat, long))
    max_x = np.amax(return_value, axis=1)[1] + para.cover_radius
    max_y = np.amax(return_value, axis=1)[2] + para.cover_radius
    min_x = np.amin(return_value, axis=1)[1] - para.cover_radius
    min_y = np.amin(return_value, axis=1)[2] - para.cover_radius
    return min_x, max_x, min_y, max_y, car_name, return_value


if __name__ == '__main__':
    read_data()
