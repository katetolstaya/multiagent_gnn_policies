from os import path
import configparser
import numpy as np
import random
import gym
import gym_flock
import torch
import sys

from learner.state_with_delay import MultiAgentStateWithDelay
from learner.gnn_dagger import DAGGER


def test(args, actor_path, k):
    # initialize gym env
    env_name = args.get('env') #'FlockingLeader-v0' #
    env = gym.make(env_name)

    debug = args.getboolean('debug')

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
    learner = DAGGER(device, args, k=k)
    n_test_episodes = args.getint('n_test_episodes')
    learner.load_model(actor_path, map_location=device)

    stats = {'mean': -1.0 * np.Inf, 'std': 0}

    test_rewards = []
    for _ in range(n_test_episodes):
        ep_reward = 0
        state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None, k=k)
        done = False
        while not done:
            action = learner.select_action(state)
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state, k=k)
            ep_reward += reward
            state = next_state
            # env.render()
        test_rewards.append(ep_reward)

        if debug:
            print(ep_reward)

    stats['mean'] = np.mean(test_rewards)
    stats['std'] = np.std(test_rewards)
    env.close()
    return stats


def main():

    # # fname = sys.argv[1]
    # base_actor_path = 'models/ddpg_actor_FlockingRelative-v0_transfer'
    # k = 3
    # fname = 'cfg/n_twoflocks.cfg'


    # actor_path = 'models/ddpg_actor_FlockingStochastic-v0_stoch2'
    # k = 2
    # fname = 'cfg/dagger_stoch.cfg'


    # base_actor_path = 'models/ddpg_actor_FlockingStochastic-v0_transfer2_stoch'
    base_actor_path = 'models/ddpg_actor_FlockingAirsimAccel-v0_transfer'
    k=3
    fname = 'cfg/airsim_dagger.cfg'

    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)

    printed_header = False

    if config.sections():
        for section_name in config.sections():
            if not printed_header:
                print(config[section_name].get('header'))
                printed_header = True

            k = config[section_name].getint('k')
            actor_path = base_actor_path + str(k)
            stats = test(config[section_name], actor_path, k=k)
            print(section_name + ", " + str(stats['mean']) + ", " + str(stats['std']))
    else:
        actor_path = base_actor_path + str(k)
        stats = test(config[config.default_section], actor_path, k=k)
        print(str(stats['mean']) + ", " + str(stats['std']))



if __name__ == "__main__":
    main()