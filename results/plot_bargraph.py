import matplotlib
import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import OrderedDict

font = {'family': 'serif',
        'weight': 'bold',
        'size': 14}
matplotlib.rc('font', **font)

_CENTRALIZED = 'Centr.'
_DECENTRALIZED = 'Decentr.'

def main():

    # fig_fname = 'airsim_trained'
    # fnames = ['airsim_trained.csv']

    fig_fname = 'stoch_transfer_to_airsim'
    fnames = ['stoch_transfer_to_airsim.csv']


    # colors = {_CENTRALIZED: 'green', _DECENTRALIZED: 'red', '4': 'blue', '3': 'darkviolet', '2': 'orange', '1': 'gold'}
    save_dir = 'fig/'

    # fnames = ['rad.csv', 'rad_baseline.csv']
    # xlabel = 'Comm. Radius'
    k_ind = 0


    mean_costs, std_costs = get_dict(fnames, k_ind)

    # max_val, min_dec = get_max(mean_costs)
    # max_val = max_val + 10.0
    ylabel = 'Cost'
    # title = ylabel + ' vs. ' + xlabel

    # plot
    fig, ax = plt.subplots()

    ax.bar(mean_costs.keys(),  mean_costs.values(), yerr = std_costs.values())


    # ax.legend()
    plt.title('Trained on Stochastic Point-Mass Model')
    # plt.ylim(top=max_val, bottom=0)
    plt.xlabel('K')
    plt.ylabel(ylabel)

    plt.savefig(save_dir + fig_fname + '.eps', format='eps')
    plt.show()


def get_dict(fnames, k_ind):
    mean_costs = OrderedDict()
    std_costs = OrderedDict()

    for fname in fnames:
        with open(fname, 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            next(plots, None)
            for row in plots:

                if len(row) == 3:
                    k = row[k_ind].strip()
                    mean = float(row[1]) * -1.0
                    std = float(row[2])

                    mean_costs[k] = mean
                    std_costs[k] = std

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
