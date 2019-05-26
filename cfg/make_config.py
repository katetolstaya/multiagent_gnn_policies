import itertools

default_fname = "default_baseline.cfg"

out_fname = 'rad_baseline.cfg'

out_file = open(out_fname, "w")

with open(default_fname) as f:
    for line in f:
        out_file.write(line)

out_file.write('\n')

params = {}
params['centralized'] = ['True','False']
params['seed'] = range(10)
# params['n_agents'] = [20, 40, 80, 100]
params['comm_radius'] = [3.0, 2.0, 1.5, 1.0]
# params['v_max'] = [0.5, 1.0, 2.0, 3.0]

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
