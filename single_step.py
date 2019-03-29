import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
import gym_flock

def train_model(env, theta, sigma, common=True, step_size=0.00001):

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

        if step % 20 == 0:
            grad = np.zeros((n_features, n_actions))

            if not common:
                for i in range(n_agents):
                    grad = grad + (action[i, :]-pi_s[i, :]).reshape(1, n_actions) * (state[i, :]).reshape(n_features, 1) * costs[i]
            else:

                avg_cost = np.sum(costs) /n_agents
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


def baseline(env, centralized):
    # test
    state = env.reset()
    avg_reward = 0
    while True:
        action = env.env.controller(centralized) 
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
parser.add_argument('--seed', type=int, default=10, metavar='N',
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
n_features = 8
n_actions = 1
sigma = 1.0

theta1 =  (np.random.rand(n_features, n_actions) * 2 - 1)
theta2 = np.copy(theta1)

baselines = []
baselines0 = []
rewards1 = []
rewards2 = []
eps = []

plt.ion()
fig, ax = plt.subplots(facecolor='white')
line, = ax.plot([], [], linewidth=2, color='g')
line0, = ax.plot([], [], linewidth=2, color='r')
line2, = ax.plot([], [], linewidth=2, color='b')
line1, = ax.plot([], [], linewidth=2, color='k')

ax.set_xlim([0, 10000])
ax.set_ylim([-8000, 0])
plt.legend((line, line0, line1, line2), ('optimal', 'consensus', 'global reward', 'local reward'))
plt.ylabel('test reward')
plt.xlabel('training episodes')

print("Optimal\tConsensus\tCommon\tLocal")
step_size=0.0002

for i_episode in range(args.num_episodes):
    #step_size = step_size * 0.99

    theta1 = train_model(env, theta1, sigma, common=True,step_size=step_size)
    #theta2 = train_model(env, theta2, sigma, common=False, step_size=step_size)

    if i_episode % 10 == 0:
        seed_state = np.random.get_state()

        baseline_reward = int(baseline(env, True))

        if seed_state is not None:
            np.random.set_state(seed_state)
        baseline0_reward = int(baseline(env, False))

        if seed_state is not None:
            np.random.set_state(seed_state)
        reward1 = int(test_model(env,theta1))

        if seed_state is not None:
            np.random.set_state(seed_state)
        reward2 = 0 #int(test_model(env,theta2))

        baselines.append(baseline_reward)
        baselines0.append(baseline0_reward)
        rewards1.append(reward1)
        rewards2.append(reward2)
        eps.append(i_episode)

        line.set_xdata(eps)
        line.set_ydata(baselines)

        line0.set_xdata(eps)
        line0.set_ydata(baselines0)
        line2.set_xdata(eps)
        line2.set_ydata(rewards2)

        line1.set_xdata(eps)
        line1.set_ydata(rewards1)



        fig.canvas.draw()
        fig.canvas.flush_events()


        print(str(baseline_reward) + "\t" +  str(baseline0_reward) + "\t" + str(reward1) + "\t" + str(reward2))



