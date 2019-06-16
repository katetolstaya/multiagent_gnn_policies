import numpy as np


def train_baseline(env, args):
    debug = args.getboolean('debug')
    n_train_episodes = args.getint('n_train_episodes')
    test_interval = args.getint('test_interval')
    n_test_episodes = args.getint('n_test_episodes')
    centralized = args.getboolean('centralized')
    total_numsteps = 0

    stats = {'mean': -1.0 * np.Inf, 'std': 0}
    # for i in range(n_train_episodes):
    #     # env.reset()
    #     # # episode_reward = 0
    #     # done = False
    #     # while not done:
    #     #
    #     #     action = env.env.controller(centralized)
    #     #     next_state, reward, done, _ = env.step(action)
    #     #     total_numsteps += 1
    #
    #     if i % test_interval == 0:
    #         test_rewards = []
    #         for _ in range(n_test_episodes):
    #             ep_reward = 0
    #             env.reset()
    #             done = False
    #             while not done:
    #                 action = env.env.controller(centralized)
    #                 next_state, reward, done, _ = env.step(action)
    #                 ep_reward += reward
    #                 # env.render()
    #             test_rewards.append(ep_reward)
    #
    #         mean_reward = np.mean(test_rewards)
    #         if stats['mean'] < mean_reward:
    #             stats['mean'] = mean_reward
    #             stats['std'] = np.std(test_rewards)
    #
    #         if debug:
    #             print(
    #                 "Episode: {}, total numsteps: {}, reward: {}".format(
    #                     i,
    #                     total_numsteps,
    #                     mean_reward))

    test_rewards = []
    for _ in range(n_test_episodes):
        ep_reward = 0
        env.reset()
        done = False
        while not done:
            action = env.env.controller(centralized)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            # env.render()
        test_rewards.append(ep_reward)

    mean_reward = np.mean(test_rewards)
    stats['mean'] = mean_reward
    stats['std'] = np.std(test_rewards)

    env.close()
    return stats
