from getdata import read_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_reward(path):
    data = pd.read_csv(path, header=0, index_col=None)
    running_rws = data['Reward']
    eps = range(1, data.shape[0] + 1, 1)
    plt.plot(eps, running_rws.to_numpy())
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()


def plot_cover_area(path):
    data = pd.read_csv(path, header=0, index_col=None)
    cover_area = data['Cover_area']
    sent_pkg = data['Sent_pkg']
    sharing = data['Sharing_factor']
    eps = range(1, data.shape[0] + 1, 1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle('Testing episode')
    ax1.plot(eps, (cover_area.to_numpy()))
    ax2.plot(eps, (sent_pkg.to_numpy()))
    ax3.plot(eps, (sharing.to_numpy()))
    ax1.set(xlabel='step', ylabel='cover area')
    ax2.set(xlabel='step', ylabel='sent')
    ax3.set(xlabel='step', ylabel='sharing')
    plt.show()

def plot_test():
    data = pd.read_csv('Data/test.csv', header=0, index_col=None)
    cover_area = data['cover_area'].round(decimals=2)
    sent_pkg = data['sent'].round(decimals=2)
    sharing = data['sharing'].round(decimals=2)
    reward = data['reward'].round(decimals=2)
    # ovl_area = data['ovl_area']
    # plt.locator_params(axis='y', nbins=3)
    eps = range(1, data.shape[0] + 1, 1)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
    fig.suptitle('Testing episode')
    ax1.plot(eps[:10], (cover_area.to_numpy()[:10]))
    # ax2.plot(eps[:100], (ovl_area.to_numpy()[:100]))
    ax3.plot(eps[:100], (reward.to_numpy()[:100]))
    ax4.plot(eps[:100], (sent_pkg.to_numpy()[:100]))
    ax5.plot(eps[:100], (sharing.to_numpy()[:100]))
    ax1.set(ylabel='cover area')
    # ax2.set(ylabel='overlap area')
    ax3.set(ylabel='reward')
    ax4.set(ylabel='sent')
    ax5.set(xlabel='step', ylabel='sharing')
    plt.show()

def plot_map():
    inputfile = "input/carname.txt"
    total_node, nodes, min_x, max_x, min_y, max_y, _ = read_data(inputfile)
    for node_i in nodes:
        plt.plot(node_i.location_list[1, :], node_i.location_list[2, :])

    plt.show()

def plot_circle():
    inputfile = "input/carname.txt"
    total_node, nodes, min_x, max_x, min_y, max_y, radi = read_data(inputfile)
    import matplotlib.patches as pt
    for node_i in nodes:
        cir = pt.Circle((node_i.location_list[1, 0], node_i.location_list[2, 0]), radius=radi, fill=False)
        plt.gcf().gca().add_artist(cir)
    plt.show()

if __name__ == '__main__':
    # plot_reward('Data/data_20210806-2101.csv')
    # plot_cover_area('Data/data_20210806-2101.csv')
    # plot_test()
    # plot_map()
    plot_circle()
