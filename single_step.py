import argparse
import gym
import numpy as np
from gym import wrappers
import gym_flock

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="FlockingMulti-v0",
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--param_noise', type=bool, default=False)
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')
parser.add_argument('--num_steps', type=int, default=200, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=100000, metavar='N',
                    help='number of episodes (default: 1000)')


args = parser.parse_args()
np.random.seed(args.seed)

env = gym.make(args.env_name)

state = env.reset()
n_agents = 15
n_features = 18
n_actions = 2
sigma = 0.01

pi_w = np.random.rand(n_features, n_actions) * 2 - 1

avg_avg_reward = 0

for i_episode in range(args.num_episodes):

    state = env.reset()
    step = 0

    avg_reward = 0

    while True:

        state = state.reshape(n_agents, n_features)
        pi_s = state.dot(pi_w)
        action = pi_s + np.random.normal(0, sigma, size=(n_agents, 2))
        next_state, reward, done, _ = env.step(action.flatten())
        #next_state, costs = next_state

        grad = np.zeros((n_features, n_actions))

        for i in range(n_agents):
            grad = grad + (action[i, :]-pi_s[i, :]).reshape(1, n_actions) * (state[i, :]).reshape(n_features, 1) #* costs[i]

        pi_w = pi_w - 0.001 * grad * reward

        state = next_state

        avg_reward = avg_reward + reward

        if done:
            break

    avg_avg_reward = avg_avg_reward + avg_reward/200
    if i_episode % 100 == 0:
        print(avg_avg_reward/100)
        avg_avg_reward = 0


