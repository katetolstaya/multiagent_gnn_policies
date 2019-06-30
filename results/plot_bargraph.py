import matplotlib
import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import OrderedDict

font = {'family': 'serif',
        'weight': 'bold',
        'size': 12}
matplotlib.rc('font', **font)

_CENTRALIZED = 'Centr.'
_DECENTRALIZED = 'Decentr.'

def main():

    # fig_fname = 'airsim_trained'

    fig_fname = 'airsim_test'


    save_dir = 'fig/'

    k_ind = 0

    mean_cost_cent = [2.041724997601464]
    std_cost_cent = [0.07619242768446846]

    mean_cost_decent = [9.335155311869324]
    std_cost_decent = [2.6593089180772074]


    mean_costs_airsim, std_costs_airsim = get_dict(['airsim_trained2.csv'], k_ind)
    mean_costs_stoch, std_costs_stoch = get_dict(['stoch_transfer_to_airsim2.csv'], k_ind)
    ylabel = 'Cost'

    # plot
    fig, ax = plt.subplots()

    ind = np.array(range(1,5))
    width = 0.35/2  # the width of the bars
    # p1 = ax.bar(ind, menMeans, width, bottom=0 * cm, yerr=menStd)
    # p2 = ax.bar(ind + width, womenMeans, width, bottom=0 * cm, yerr=womenStd)

    p1 = ax.bar(ind-width,  mean_costs_airsim.values(), width=width*2, yerr = std_costs_airsim.values())
    p2 = ax.bar(ind+width,  mean_costs_stoch.values(), width=width*2, yerr = std_costs_stoch.values())
    p3 = ax.bar(0, mean_cost_cent, width=width*3, yerr=std_cost_cent)
    p4 = ax.bar(-1, mean_cost_decent, width=width*3,  yerr=std_cost_decent)


    ax.legend((p1[0], p2[0], p3[0], p4[0]), ('Trained: AirSim', 'Trained: Point-Masses', 'Global', 'Local'))
    plt.title('Testing in AirSim')
    # plt.ylim(top=max_val, bottom=0)
    plt.xlabel('K')
    plt.ylabel(ylabel)

    ax.set_xticklabels(('', '', '', '1', '2', '3', '4'))

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


if __name__ == "__main__":
    main()
