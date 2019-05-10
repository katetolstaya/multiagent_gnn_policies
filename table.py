from gnn_dagger import train_dagger
import gym
import gym_flock
import random
import numpy as np
import torch

import argparse

''' Parse Arguments'''
parser = argparse.ArgumentParser(description='DAGGER Implementation')

# learning parameters
parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
parser.add_argument('--buffer_size', type=int, default=10000, help='Replay Buffer Size')
parser.add_argument('--updates_per_step', type=int, default=200, help='Updates per Batch')
parser.add_argument('--seed', type=int, default=11, help='random_seed')
parser.add_argument('--actor_lr', type=float, default=5e-5, help='learning rate for actor')

# architecture parameters
parser.add_argument('--k', type=int, default=2, help='k')
parser.add_argument('--hidden_size', type=int, default=32, help='hidden layer size')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
parser.add_argument('--tau', type=float, default=0.5, help='tau')

# env parameters
parser.add_argument('--env', type=str, default="FlockingRelative-v0", help='Gym environment to run')
parser.add_argument('--v_max', type=float, default=3.0, help='maximum initial flock vel')
parser.add_argument('--comm_radius', type=float, default=1.0, help='flock communication radius')
parser.add_argument('--n_agents', type=int, default=90, help='n_agents')
parser.add_argument('--n_actions', type=int, default=2, help='n_actions')
parser.add_argument('--n_states', type=int, default=6, help='n_states')

args = parser.parse_args()


def main():
    # initialize gym env
    env = gym.make(args.env)

    if args.env == "FlockingRelative-v0":
        env.env.set_num_agents(args.n_agents)
        env.env.set_comm_radius(args.comm_radius)
        env.env.set_initial_vmax(args.v_max)

    print("k, reward")
    for k in range(1, 6):

        # use seed
        env.seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        args.k = k

        # initialize params tuple
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        best_avg_reward = train_dagger(env, args, device, False)

        print("{}, {}".format(k, best_avg_reward))


if __name__ == "__main__":
    main()
