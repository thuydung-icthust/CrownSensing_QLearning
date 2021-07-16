from getdata import read_data
import matplotlib.pyplot as plt
import pandas as pd


def plot_reward(path):
    data = pd.read_csv(path, header=0, index_col=None)
    running_rws = data['Reward']
    eps = range(1, 1001, 1)
    plt.plot(eps, running_rws.to_numpy())
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()


def plot_cover_area(path):
    data = pd.read_csv(path, header=0, index_col=None)
    cover_area = data['Cover_area']
    sent_pkg = data['Sent_pkg']
    sharing = data['Sharing_factor']
    eps = range(1, 201, 1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle('Testing episode')
    ax1.plot(eps, (cover_area.to_numpy()[:200] / 1000))
    ax2.plot(eps, (sent_pkg.to_numpy()[:200] / 1000))
    ax3.plot(eps, (sharing.to_numpy()[:200] / 1000))
    ax1.set(xlabel='step', ylabel='cover area')
    ax2.set(xlabel='step', ylabel='sent')
    ax3.set(xlabel='step', ylabel='sharing')
    plt.show()


def plot_map():
    inputfile = "input/carname.txt"
    total_node, nodes, min_x, max_x, min_y, max_y = read_data(inputfile)
    for node_i in nodes:
        plt.plot(node_i.location_list[1, :], node_i.location_list[2, :])

    plt.show()


if __name__ == '__main__':
    # plot_reward('log/data_20210709-2259.csv')
    # plot_cover_area('Data/data_20210712-2045.csv')
    plot_map()
