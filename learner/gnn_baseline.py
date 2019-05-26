import numpy as np


def train_baseline(env, args):
    debug = args.getboolean('debug')
    n_train_episodes = args.getint('n_train_episodes')
    test_interval = args.getint('test_interval')
    n_test_episodes = args.getint('n_test_episodes')
    centralized = args.getboolean('centralized')

    rewards = []
    total_numsteps = 0

    for i in range(n_train_episodes):
        env.reset()

        # episode_reward = 0
        done = False
        while not done:

            action = env.env.controller(centralized)
            next_state, reward, done, _ = env.step(action)
            total_numsteps += 1

        if i % test_interval == 0:
            episode_reward = 0
            for _ in range(n_test_episodes):
                env.reset()
                done = False
                while not done:
                    action = env.env.controller(centralized)
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward

            rewards.append(episode_reward/n_test_episodes)

            if debug:
                print(
                    "Episode: {}, total numsteps: {}, reward: {}".format(
                        i,
                        total_numsteps,
                        rewards[-1]))

    env.close()
    return np.max(rewards)
