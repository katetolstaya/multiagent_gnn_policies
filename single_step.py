import argparse
import gym
import numpy as np
from gym import wrappers
import gym_flock

def train_model(env, theta, sigma, common=True, step_size=0.0001):

    # train
    state = env.reset()
    step = 0

    avg_reward = 0
    step = 0

    while True:
        state = state.reshape(n_agents, n_features)

        pi_s = state.dot(theta)
        action = pi_s + np.random.normal(0, sigma, size=(n_agents, n_actions))
        next_state, reward, done, _ = env.step(action.flatten())
        next_state, costs = next_state

        if step % 10 == 0:
            grad = np.zeros((n_features, n_actions))

            if not common:
                for i in range(n_agents):
                    grad = grad + (action[i, :]-pi_s[i, :]).reshape(1, n_actions) * (state[i, :]).reshape(n_features, 1) * costs[i]
            else: 
                avg_cost = np.sum(costs) #/n_agents
                for i in range(n_agents):
                    grad = grad + (action[i, :]-pi_s[i, :]).reshape(1, n_actions) * (state[i, :]).reshape(n_features, 1) * avg_cost
 
            theta = theta + step_size * grad 

        state = next_state
        step = step + 1

        if done:
            break

    return theta


def test_model(env, weights):
    # test
    state = env.reset()
    avg_reward = 0
    while True:
        state = state.reshape(n_agents, n_features)
        action = state.dot(weights)
        next_state, reward, done, _ = env.step(action.flatten())
        next_state, costs = next_state
        state = next_state
        avg_reward = avg_reward + reward

        if done:
            break
    return avg_reward


def baseline(env):
    # test
    state = env.reset()
    avg_reward = 0
    while True:
        action = env.env.controller() 
        next_state, reward, done, _ = env.step(action.flatten())
        avg_reward = avg_reward + reward
        if done:
            break
    return avg_reward

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="Consensus-v0",
                    help='name of the environment to run')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--param_noise', type=bool, default=False)
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=43, metavar='N',
                    help='random seed (default: 4)')
parser.add_argument('--num_steps', type=int, default=500, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=100000, metavar='N',
                    help='number of episodes (default: 1000)')


args = parser.parse_args()
np.random.seed(args.seed)

env = gym.make(args.env_name)

state = env.reset()
n_agents = 50
# n_features = 18
# n_actions = 2
n_features = 6
n_actions = 1
sigma = 30.0

theta1 =  (np.random.rand(n_features, n_actions) * 2 - 1)
theta2 = np.copy(theta1)

print("Baseline\tCommon\tLocal")
for i_episode in range(args.num_episodes):

    #sigma = sigma * 0.99

    theta1 = train_model(env, theta1, sigma, common=True)
    theta2 = train_model(env, theta2, sigma, common=False)

    if i_episode % 10 == 0:
        print(str(round(baseline(env))) + "\t" + str(round(test_model(env,theta1))) + "\t" + str(round(test_model(env,theta2))))



