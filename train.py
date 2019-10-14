from os import path
import configparser
import numpy as np
import random
import gym
import gym_flock
import torch
import sys

from learner.gnn_cloning import train_cloning
from learner.gnn_dagger import train_dagger
from learner.gnn_baseline import train_baseline
from learner.gnn_pac import train_dagger as train_pac_dagger
from learner.gnn_pac_ddpg import train as train_gnn_pac_ddpg


def run_experiment(args):
    # initialize gym env
    env_name = args.get('env')
    env = gym.make(env_name)

    if isinstance(env.env, gym_flock.envs.FlockingRelativeEnv) or isinstance(env.env, gym_flock.envs.MappingLocalEnv):
        env.env.params_from_cfg(args)

    # use seed
    seed = args.getint('seed')
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if 'device' in args:
        device_string = args.get('device')
    else:
        device_string = "cuda:1"

    # initialize params tuple
    device = torch.device(device_string if torch.cuda.is_available() else "cpu")
    print(device)

    alg = args.get('alg').lower()
    if alg == 'dagger':
        stats = train_dagger(env, args, device)
    elif alg == 'cloning':
        stats = train_cloning(env, args, device)
    elif alg == 'baseline':
        stats = train_baseline(env, args)
    elif alg == 'pac':
        stats = train_pac_dagger(env, args, device)
    elif alg == 'gnn_pac_ddpg':
        stats = train_gnn_pac_ddpg(env, args, device)
    else:
        raise Exception('Invalid algorithm/mode name')
    return stats


def main():
    fname = sys.argv[1]
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)

    printed_header = False

    if config.sections():
        for section_name in config.sections():
            if not printed_header:
                print(config[section_name].get('header'))
                printed_header = True

            stats = run_experiment(config[section_name])
            print(section_name + ", " + str(stats['mean']) + ", " + str(stats['std']))
    else:
        val = run_experiment(config[config.default_section])
        print(val)


if __name__ == "__main__":
    main()
