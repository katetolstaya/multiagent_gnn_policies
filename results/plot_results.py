import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import OrderedDict

fname = 'vel2.csv'
xlabel = 'Max Initial Velocity'
ylabel = 'Avg Cost'
k_ind = 1
v_ind = 0


fname = 'rad2.csv'
xlabel = 'Comm. Radius'
ylabel = 'Avg Cost'
k_ind = 0
v_ind = 2


title = ylabel + ' vs. ' + xlabel

# construct dict for results of parameter sweep
list_costs = OrderedDict()
with open(fname, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
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

# compute average over diff seeds for each parameter combo
avg_costs = OrderedDict()
for k in list_costs.keys():
    if k not in avg_costs:
        avg_costs[k] = OrderedDict()

    for v in list_costs[k].keys():
        avg_costs[k][v] = np.mean(list_costs[k][v])

# plot
fig, ax = plt.subplots()
for k in avg_costs.keys():
    ax.plot(avg_costs[k].keys(), avg_costs[k].values(), marker='o', label='K=' + str(k))

ax.legend()
plt.title(title)

plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.show()
