import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import OrderedDict


def get_dict(fname, k_ind, v_ind):
    list_costs = OrderedDict()
    with open(fname, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots, None)
        for row in plots:
            if len(row) == 4:
                k = int(row[k_ind])
                v = float(row[v_ind])
                reward = float(row[3])
                cost = -1.0 * reward

                if k not in list_costs:
                    list_costs[k] = OrderedDict()

                if v not in list_costs[k]:
                    list_costs[k][v] = []

                list_costs[k][v].append(cost)
    return list_costs

def get_dict_baseline(fname, k_ind, v_ind):
    list_costs = OrderedDict()
    with open(fname, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots, None)
        for row in plots:
            if len(row) == 4:
                k = row[k_ind]
                k = k.replace(" ", "")
                if k == 'True':
                    k = 'Centralized'
                else:
                    k = 'Decentralized'

                v = float(row[v_ind])
                reward = float(row[3])
                cost = -1.0 * reward

                if k not in list_costs:
                    list_costs[k] = OrderedDict()

                if v not in list_costs[k]:
                    list_costs[k][v] = []

                list_costs[k][v].append(cost)
    return list_costs

def get_mean(list_costs):
    # compute average over diff seeds for each parameter combo
    avg_costs = OrderedDict()
    for k in list_costs.keys():
        if k not in avg_costs:
            avg_costs[k] = OrderedDict()

        for v in list_costs[k].keys():
            avg_costs[k][v] = np.mean(list_costs[k][v])
    return avg_costs

def get_stddev(list_costs):
    # compute standard deviation
    std_costs = OrderedDict()
    for k in list_costs.keys():
        if k not in std_costs:
            std_costs[k] = OrderedDict()

        for v in list_costs[k].keys():
            std_costs[k][v] = np.std(list_costs[k][v])

    return std_costs

# fname_baseline = 'vel4_baseline.csv'
# k_ind_baseline = 2
# v_ind_baseline = 0
#
# fname = 'vel2.csv'
# xlabel = 'Max Initial Velocity'
# ylabel = 'Avg Cost'
# k_ind = 1
# v_ind = 0

fname_baseline = 'rad4_baseline.csv'
k_ind_baseline = 2
v_ind_baseline = 0

fname = 'rad4.csv'
xlabel = 'Comm. Radius'
ylabel = 'Avg Cost'
k_ind = 0
v_ind = 2


title = ylabel + ' vs. ' + xlabel

list_costs = get_dict(fname, k_ind, v_ind)
avg_costs = get_mean(list_costs)
std_costs = get_stddev(list_costs)

list_costs_baseline = get_dict_baseline(fname_baseline, k_ind_baseline, v_ind_baseline)
avg_costs_baseline = get_mean(list_costs_baseline)
std_costs_baseline = get_stddev(list_costs_baseline)

# plot
fig, ax = plt.subplots()
for k in avg_costs.keys():
    ax.errorbar(avg_costs[k].keys(), avg_costs[k].values(), yerr=std_costs[k].values(), marker='o', label='K=' + str(k))

for k in avg_costs_baseline.keys():

    ax.errorbar(avg_costs_baseline[k].keys(), avg_costs_baseline[k].values(), yerr=std_costs_baseline[k].values(),
                marker='o', label=k)


ax.legend()
plt.title(title)

plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.show()
