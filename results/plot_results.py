import matplotlib
import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import OrderedDict

font = {'family': 'serif',
        'weight': 'bold',
        'size': 14}
matplotlib.rc('font', **font)

_CENTRALIZED = 'Global'
_DECENTRALIZED = 'Local'

def main():

    fig_fname = 'n'

    if fig_fname == 'transfer_grid_n':
        fnames = ['transfer_grid_n.csv']
        xlabel = 'Number of Agents'
        k_ind = 0
        v_ind = 1
        arrow_params = None

    if fig_fname == 'transfer_twoflocks_n':
        fnames = ['transfer_twoflocks_n.csv']
        xlabel = 'Number of Agents'
        k_ind = 0
        v_ind = 1
        arrow_params = None

    if fig_fname == 'vel':
        fnames = ['vel.csv', 'vel_baseline.csv']
        xlabel = 'Maximum Initial Velocity'
        k_ind = 1
        v_ind = 0

        arrow_params = {'x':3.3, 'y': 400.0, 'dx':0.0, 'dy':30.0, 'width':0.03, 'head_length':30, 'color':'r'}
        text_params = {'x': 2.7, 'y': 370}

    if fig_fname == 'transfer_vel':
        fnames = ['transfer_vel.csv', 'vel_baseline.csv']
        xlabel = 'Maximum Initial Velocity'
        k_ind = 1
        v_ind = 0

        arrow_params = None

        # arrow_params = {'x':70, 'y': 400.0, 'dx':0.0, 'dy':30.0, 'width':1.5, 'head_length':30, 'color':'r'}
        # text_params = {'x': 55, 'y': 370}

    if fig_fname == 'transfer_n':
        fnames = ['transfer_n.csv', 'n_baseline.csv']
        xlabel = 'Number of Agents'
        k_ind = 0
        v_ind = 1

        arrow_params = {'x':105, 'y': 230.0, 'dx':0.0, 'dy':25.0, 'width':1.5, 'head_length':30, 'color':'r'}
        text_params = {'x': 85, 'y': 200}

    if fig_fname == 'transfer_leader_vel':
        fnames = ['transfer_leader_vel.csv', 'vel_leader_baseline.csv']
        xlabel = 'Maximum Initial Velocity'
        k_ind = 1
        v_ind = 0
        arrow_params = None

    if fig_fname == 'rad':
        fnames = ['rad.csv', 'rad_baseline.csv']
        xlabel = 'Comm. Radius'
        k_ind = 0
        v_ind = 1

        arrow_params = None

    if fig_fname == 'n':
        fnames = ['n.csv', 'n_baseline.csv']
        xlabel = 'Number of Agents'
        k_ind = 0
        v_ind = 1

        arrow_params = {'x':85, 'y': 300.0, 'dx':0.0, 'dy':30.0, 'width':1.5, 'head_length':30, 'color':'r'}
        text_params = {'x': 70, 'y': 370}

    colors = {_CENTRALIZED: 'green', _DECENTRALIZED: 'red', '4': 'blue', '3': 'darkviolet', '2': 'orange', '1': 'gold'}
    save_dir = 'fig/'

    mean_costs, std_costs = get_dict(fnames, k_ind, v_ind)

    max_val, min_dec = get_max(mean_costs)
    max_val = max_val + 10.0
    ylabel = 'Cost'
    title = ylabel + ' vs. ' + xlabel

    # plot
    fig, ax = plt.subplots()
    for k in mean_costs.keys():
        # if k != '4':
        if not (k == _CENTRALIZED or k == _DECENTRALIZED):
            label = 'K=' + k
        else:
            label = k
        ax.errorbar(mean_costs[k].keys(), mean_costs[k].values(), yerr=std_costs[k].values(), marker='o', color=colors[k],
                    label=label)

    ax.legend()
    plt.title(title)
    plt.ylim(top=max_val, bottom=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if max_val < min_dec < np.Inf and arrow_params:
        min_dec = int(np.floor(min_dec / 100.0)*100)
        # plt.arrow(x=3.3, y=400.0, dx=0.0, dy=30.0, color='r', width=0.03, head_length=30)
        plt.arrow(**arrow_params)
        plt.text(x=text_params['x'], y=text_params['y'], s='Cost > '+str(min_dec), color='r')

    plt.savefig(save_dir + fig_fname + '.eps', format='eps')
    plt.show()


def get_dict(fnames, k_ind, v_ind):
    mean_costs = OrderedDict()
    std_costs = OrderedDict()

    for fname in fnames:
        with open(fname, 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            next(plots, None)
            for row in plots:

                if True: # len(row) == 4:
                    k = row[k_ind].strip()

                    if k == 'True':
                        k = _CENTRALIZED
                    elif k == 'False':
                        k = _DECENTRALIZED

                    v = float(row[v_ind])

                    mean = float(row[2]) * -1.0
                    std = float(row[3])
                    if k not in mean_costs:
                        mean_costs[k] = OrderedDict()
                        std_costs[k] = OrderedDict()
                    mean_costs[k][v] = mean
                    std_costs[k][v] = std

    return mean_costs, std_costs


def get_max(list_costs):
    # compute average over diff seeds for each parameter combo
    max_val = -1.0 * np.Inf
    min_decentralized = 1.0 * np.Inf

    for k in list_costs.keys():
        for v in list_costs[k].keys():
            if k != _DECENTRALIZED:
                max_val = np.maximum(max_val, list_costs[k][v])
            else:
                min_decentralized = np.minimum(min_decentralized, list_costs[k][v])
    return max_val, min_decentralized

if __name__ == "__main__":
    main()
