from os import path
import configparser
import numpy as np
import random
import gym
import gym_flock
import torch

from gnn_cloning import train_cloning
from gnn_dagger import train_dagger


def run_experiment(args):
    # initialize gym env
    env_name = args.get('env')
    env = gym.make(env_name)

    if env_name == "FlockingRelative-v0":
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


def main():
    config_file = path.join(path.dirname(__file__), "cfg/vel.cfg")
    config = configparser.ConfigParser()
    config.read(config_file)

    if config.sections():
        for section_name in config.sections():
            val = run_experiment(config[section_name])
            print(section_name + ", " + str(val))
    else:
        val = run_experiment(config[config.default_section])
        print(val)


if __name__ == "__main__":
    main()
