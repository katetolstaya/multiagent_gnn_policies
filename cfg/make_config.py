import itertools

baseline = False
# baseline = True
# param = 'vel'
# param = 'dt'
# param = 'n'
# param = 'rad'

param = 'hidden_size'

params = {}
#
# params['seed'] = range(10)

if baseline:
    default_fname = "default_baseline.cfg"
    out_fname = param + '_baseline.cfg'

    params['centralized'] = ['True', 'False']
else:
    default_fname = 'default.cfg'
    out_fname = param + '.cfg'
    # params['n_layers'] = [1, 2, 3, 4]

    params['k'] = [1, 2, 3, 4]

if param == 'vel':
    params['v_max'] = [0.5, 1.5, 2.5, 3.5, 4.5]
elif param == 'rad':
    params['comm_radius'] = [3.0, 2.5, 2.0, 1.5, 1.0]
elif param == 'n':
    params['n_agents'] = [25, 50, 75, 100, 125, 150, 175, 200]
elif param == 'dt':
    params['dt'] = [0.1, 0.075, 0.05, 0.025, 0.01, 0.0075]
elif param == 'hidden_size':
    params['hidden_size'] = [4, 8, 16, 32, 64, 128]


out_file = open(out_fname, "w")

with open(default_fname) as f:
    for line in f:
        out_file.write(line)

out_file.write('\n')

param_names = params.keys()
param_values = params.values()

line = ''
for name in param_names:
    line = line + name + ', '

out_file.write('header = ' + line + 'reward' '\n\n')

for element in itertools.product(*param_values):
    line = '['
    for v in element:
        line = line + str(v) + ', '
    line = line[0:-2] + ']'
    out_file.write(line + '\n')

    for i in range(len(param_names)):
        line = param_names[i] + ' = ' + str(element[i])
        out_file.write(line + '\n')

    out_file.write('\n')

out_file.close()
