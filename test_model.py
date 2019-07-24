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


def test(args, actor_path, render=True):
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
    learner = DAGGER(device, args)
    n_test_episodes = args.getint('n_test_episodes')
    learner.load_model(actor_path, device)

    for _ in range(n_test_episodes):
        episode_reward = 0
        state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)
        done = False
        while not done:
            action = learner.select_action(state)
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state)
            episode_reward += reward
            state = next_state
            if render:
                env.render()
        print(episode_reward)
    env.close()


def main():
    fname = sys.argv[1]
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)

    printed_header = False
    actor_path = 'models/actor_FlockingRelative-v0_dagger_k3'

    if config.sections():
        for section_name in config.sections():
            if not printed_header:
                print(config[section_name].get('header'))
                printed_header = True

            test(config[section_name], actor_path)
    else:
        test(config[config.default_section], actor_path)



if __name__ == "__main__":
    main()