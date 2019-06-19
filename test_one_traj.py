from os import path
import configparser
import numpy as np
import random
import gym
import gym_flock
import torch
# import sys

from learner.state_with_delay import MultiAgentStateWithDelay
from learner.gnn_dagger import DAGGER

# import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

font = {'family': 'serif',
        'weight': 'bold',
        'size': 14}

matplotlib.rc('font', **font)

save_dir = 'results/fig/'


def test(args, actor_path, k):
    # initialize gym env
    env_name = args.get('env')
    print(env_name)
    # env_name = "FlockingAirsimAccel-v0"
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
    learner = DAGGER(device, args, k)
    n_test_episodes = args.getint('n_test_episodes')

    learner.load_model(actor_path, device)

    n_steps = 500

    steps = np.zeros((n_steps,))
    min_dists_mean = np.zeros((n_steps,))
    min_dists_std = np.zeros((n_steps,))

    vel_diffs_mean = np.zeros((n_steps,))
    vel_diffs_std = np.zeros((n_steps,))

    episode_reward = 0
    state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None, k=k)

    for step in range(0, n_steps):
        action = learner.select_action(state)
        next_state, reward, done, _ = env.step(action.cpu().numpy())

        # next_state, reward, done, _ = env.step(env.env.controller(True))

        stats = env.env.get_stats()

        min_dists_mean[step] = np.mean(stats['min_dists'])
        vel_diffs_mean[step] = np.mean(stats['vel_diffs'])
        min_dists_std[step] = np.std(stats['min_dists'])
        vel_diffs_std[step] = np.std(stats['vel_diffs'])

        steps[step] = step

        next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state, k=k)
        episode_reward += reward
        state = next_state

        if step % 100 == 0:
            env.render()
            plt.savefig(save_dir + 'traj' + str(step) + '.eps', format='eps')

    plt.ioff()

    y = min_dists_mean
    y_min = min_dists_mean - min_dists_std
    y_max = min_dists_mean + min_dists_std

    fig = plt.figure()
    plt.plot(steps, y, 'b-')
    plt.fill_between(steps, y_min, y_max, color='lightblue')
    plt.xlabel('Step')
    plt.ylabel('Min. Distances')
    plt.tight_layout()
    plt.savefig(save_dir + 'min_dist.eps', format='eps')
    plt.show()

    y = vel_diffs_mean
    y_min = vel_diffs_mean - vel_diffs_std
    y_max = vel_diffs_mean + vel_diffs_std

    fig = plt.figure()
    plt.plot(steps, y, 'r-')
    plt.fill_between(steps, y_min, y_max, color='orange')
    plt.xlabel('Step')
    plt.ylabel('Velocity Diff.')
    plt.tight_layout()
    plt.savefig(save_dir + 'vel_diff.eps', format='eps')
    plt.show()

    env.close()


def main():
    # fname = sys.argv[1]

    # actor_path = 'models/ddpg_actor_FlockingRelative-v0_k3'
    # actor_path = 'models/ddpg_actor_FlockingRelative-v0_k3'
    # actor_path = 'models/ddpg_actor_FlockingStochastic-v0_stoch2'

    k = 3
    actor_path = 'models/ddpg_actor_FlockingRelative-v0_transfer' + str(k)
    fname = 'cfg/dagger.cfg'

    # actor_path = 'models/ddpg_actor_FlockingStochastic-v0_transfer_stoch' + str(k)
    # fname = 'cfg/dagger_stoch.cfg'

    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)

    printed_header = False

    if config.sections():
        for section_name in config.sections():
            if not printed_header:
                print(config[section_name].get('header'))
                printed_header = True

            test(config[section_name], actor_path, k)
    else:
        test(config[config.default_section], actor_path, k)


if __name__ == "__main__":
    main()
