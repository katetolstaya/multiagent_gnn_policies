import gym
import gym_flock
import configparser

env_name = "FlockingAirsimAccel-v0"
env = gym.make(env_name)
config_file = 'cfg/airsim_dagger.cfg'

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
        #env.render()

    print(episode_reward)

env.close()