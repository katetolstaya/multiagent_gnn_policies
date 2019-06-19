import gym
import gym_flock
import configparser


env_name = "FlockingTwoFlocks-v0"
config_file = 'cfg/n_twoflocks.cfg'

# env_name = "FlockingRelative-v0"
# config_file = 'cfg/dagger.cfg'
#
# env_name = "FlockingAirsimAccel-v0"
# config_file = 'cfg/airsim_dagger.cfg'

# env_name = 'FlockingStochastic-v0'
# config_file = 'cfg/dagger_stoch.cfg'

env = gym.make(env_name)
config = configparser.ConfigParser()
config.read(config_file)
env.env.params_from_cfg(config[config.sections()[0]])

while True:
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = env.env.controller(False)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state
        env.render()

    print(episode_reward)

env.close()