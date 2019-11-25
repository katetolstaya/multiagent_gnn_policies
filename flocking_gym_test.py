import gym
import gym_flock
import configparser
import numpy as np


env_name = "MappingDisc-v0"
# config_file = 'cfg/n_twoflocks.cfg'

# env_name = "FlockingRelative-v0"
# config_file = 'cfg/dagger.cfg'
#
# env_name = "FlockingAirsimAccel-v0"
# config_file = 'cfg/airsim_dagger.cfg'

# env_name = 'FlockingStochastic-v0'
# config_file = 'cfg/dagger_stoch.cfg'

env = gym.make(env_name)
# config = configparser.ConfigParser()
# config.read(config_file)
# env.env.params_from_cfg(config[config.sections()[0]])
N = 1000
total_reward = 0
for _ in range(N):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = env.env.controller()
        # action = env.env.controller(False)

        next_state, reward, done, _ = env.step(action)
        # print(reward)
        episode_reward += np.sum(reward)
        state = next_state
        # env.render()
    total_reward += episode_reward
    print(episode_reward)
print('total')
print(total_reward / N)

env.close()