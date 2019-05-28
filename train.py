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


def run_experiment(args):
    # initialize gym env
    env_name = args.get('env')
    env = gym.make(env_name)

    if isinstance(env.env, gym_flock.envs.FlockingRelativeEnv):
        env.env.params_from_cfg(args)

    # use seed
    seed = args.getint('seed')
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # initialize params tuple
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    alg = args.get('alg').lower()
    if alg == 'dagger':
        return train_dagger(env, args, device)
    elif alg == 'cloning':
        return train_cloning(env, args, device)
    elif alg == 'baseline':
        return train_baseline(env, args)


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

            val = run_experiment(config[section_name])
            print(section_name + ", " + str(val))
    else:
        val = run_experiment(config[config.default_section])
        print(val)


if __name__ == "__main__":
    main()
