import argparse
import math
from collections import namedtuple
from itertools import count
from tqdm import tqdm
from tensorboardX import SummaryWriter

import gym
import numpy as np
from gym import wrappers
import gym_flock

import torch
from ddpg import DDPG
from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
# parser.add_argument('--env-name', default="LQR-v0",
#                     help='name of the environment to run')
# parser.add_argument('--algo', default='NAF',
#                     help='algorithm to use: DDPG | NAF')
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
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=200, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


env_name = 'Flocking-v0'
suffix = 900
actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix) 
critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix) 

env = NormalizedActions(gym.make(env_name))
writer = SummaryWriter()

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
agent = DDPG(args.gamma, args.tau, args.hidden_size,
                      env.observation_space.shape[0], env.action_space, device)

agent.load_model(actor_path, critic_path)

memory = ReplayMemory(args.replay_size)

ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, 
    desired_action_stddev=args.noise_scale, adaptation_coefficient=1.05, device=device) if args.param_noise else None

rewards = []
total_numsteps = 0
updates = 0

state = env.reset()
n_agents = state.shape[0]

action = np.zeros((n_agents, env.action_space.shape[0]))

for i_episode in range(args.num_episodes):

    state = env.reset() # TODO

    if args.ou_noise: 
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                      i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()

    if args.param_noise and args.algo == "DDPG":
        agent.perturb_actor_parameters(param_noise)

    episode_reward = 0
    while True:
        state = env.reset() #torch.Tensor([env.reset()]) # TODO
        episode_reward = 0
        while True:
            for n in range(n_agents):
                agent_state = torch.Tensor([state[n,:].flatten()]).to(device)
                agent_action = agent.select_action(agent_state, ounoise, param_noise)
                action[n, :] = agent_action.cpu().numpy()
            #action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action) # TODO
            episode_reward += reward

            state = next_state
            env.render()
            if done:
                break

        rewards.append(episode_reward)
        print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps, rewards[-1], np.mean(rewards[-10:])))

env.close()
