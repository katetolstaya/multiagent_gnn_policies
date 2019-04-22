import argparse
import numpy as np
import os
from collections import namedtuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

import gym

''' Parse Arguments'''
parser = argparse.ArgumentParser(description='DDPG Implementation')
parser.add_argument('--env', type=str, default="Pendulum-v0", help='Gym environment to run')
parser.add_argument('--batch_size', type=int, default=40, help='Batch size')
parser.add_argument('--buffer_size', type=int, default=10000, help='Replay Buffer Size')
parser.add_argument('--updates_per_step', type=int, default=1, help='Updates per Batch')
parser.add_argument('--n_agents', type=int, default=100, help='n_agents')

args = parser.parse_args()

Transition = namedtuple('Transition', ('state', 'action', 'done', 'next_state', 'reward', 'gso', 'next_gso'))

# TODO: how to deal with bounded/unbounded action spaces?? Should I always assume bounded actions?

class ReplayBuffer(object):
    """
    Stores training samples for RL algorithms, represented as tuples of (S, A, R, S).
    """

    def __init__(self, max_size=1000):
        self.buffer = []
        self.max_size = max_size
        self.curr_size = 0

    def insert(self, sample):
        """
        Insert sample into buffer.
        :param sample: The (S,A,R,S) tuple.
        :return: None
        """
        if self.curr_size == self.max_size:
            self.buffer.pop(0)
        else:
            self.curr_size += 1
        self.buffer.append(sample)

    def sample(self, num):
        """
        Sample a number of transitions from the replay buffer.
        :param num: Number of transitions to sample.
        :return: The set of sampled transitions.
        """
        return random.sample(self.buffer, num)

    def clear(self):
        """
        Clears the current buffer.
        :return: None
        """
        self.buffer = []
        self.curr_size = 0

# TODO move these generic implementations to a different file for later
class GNN(nn.Module):

    def __init__(self, layers, K):
        super(GNN, self).__init__()

        self.layers = layers
        self.n_layers = len(layers) - 1
        self.conv_layers = []

        for i in range(self.n_layers):
            m = nn.Conv1d(in_channels=K, out_channels=layers[i + 1], kernel_size=(layers[i], 1), stride=(layers[i], 1))
            self.conv_layers.append(m)
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        # self.activation = nn.Tanh()

    def forward(self, inputs, graph_shift_ops):
        batch_size = np.shape(inputs)[0]
        x = inputs
        for i in range(self.n_layers - 1):
            x = torch.matmul(x, graph_shift_ops)
            # x = self.activation(self.conv_layers[i](x))
            x = F.relu(self.conv_layers[i](x))
            x = x.view((batch_size, 1, self.layers[i + 1], -1))
        x = self.conv_layers[self.n_layers](x)

        return x  # size of x here is batch_size x output_layer x 1 x n_agents


class DelayGNN(nn.Module):

    def __init__(self, layers, K):
        super(DelayGNN, self).__init__()

        self.layers = layers
        self.n_layers = len(layers) - 1

        self.conv_layers = []
        m = nn.Conv1d(in_channels=K, out_channels=layers[1], kernel_size=(layers[0], 1), stride=(layers[0], 1))
        self.conv_layers.append(m)

        for i in range(1, self.n_layers):
            m = nn.Conv1d(in_channels=1, out_channels=layers[i + 1], kernel_size=(layers[i], 1), stride=(layers[i], 1))
            self.conv_layers.append(m)

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

    def forward(self, inputs, graph_shift_ops):
        x = inputs
        x = torch.matmul(x, graph_shift_ops)

        for i in range(self.n_layers - 1):
            x = F.relu(self.conv_layers[i](x))
            # x = x.view((-1, 1, self.layers[i + 1], self.n_agents))  # necessary?

        x = self.conv_layers[self.n_layers](x)

        return x  # size of x here is batch_size x output_layer x 1 x n_agents


class Critic(nn.Module):

    def __init__(self, layers, K):
        super(Critic, self).__init__()

        self.layers = layers
        self.n_layers = len(layers) - 1
        self.conv_layers = []

        for i in range(self.n_layers):
            m = nn.Conv1d(in_channels=K, out_channels=layers[i + 1], kernel_size=(layers[i], 1), stride=(layers[i], 1))
            self.conv_layers.append(m)
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        # self.activation = nn.Tanh()

    def forward(self, states, actions, graph_shift_ops):
        batch_size = np.shape(states)[0]
        x = torch.cat((states, actions), 2)
        for i in range(self.n_layers - 1):
            x = torch.matmul(x, graph_shift_ops)
            # x = self.activation(self.conv_layers[i](x))
            x = F.relu(self.conv_layers[i](x))
            x = x.view((batch_size, 1, self.layers[i + 1], -1))
        x = self.conv_layers[self.n_layers](x)

        return x  # size of x here is batch_size x output_layer x 1 x n_agents


class Actor(nn.Module):

    def __init__(self, layers, K):
        super(Actor, self).__init__()

        self.layers = layers
        self.n_layers = len(layers) - 1

        self.conv_layers = []
        m = nn.Conv1d(in_channels=K, out_channels=layers[1], kernel_size=(layers[0], 1), stride=(layers[0], 1))
        self.conv_layers.append(m)

        for i in range(1, self.n_layers):
            m = nn.Conv1d(in_channels=1, out_channels=layers[i + 1], kernel_size=(layers[i], 1), stride=(layers[i], 1))
            self.conv_layers.append(m)

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

    def forward(self, states, graph_shift_ops):
        x = states
        x = torch.matmul(x, graph_shift_ops)

        for i in range(self.n_layers - 1):
            x = F.relu(self.conv_layers[i](x))
            # x = x.view((-1, 1, self.layers[i + 1], self.n_agents))  # necessary?

        x = self.conv_layers[self.n_layers](x)

        return x  # size of x here is batch_size x output_layer x 1 x n_agents



class OUNoise:
    """
    Generates noise from an Ornstein Uhlenbeck process, for temporally correlated exploration. Useful for physical
    control problems with inertia. See https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """
    def __init__(self, nA, N, scale=0.3, mu=0, theta=0.15, sigma=0.2):
        """
        Initialize the Noise parameters.
        :param nA: Size of the Action space.
        :param scale: Scale of the noise process.
        :param mu: The mean of the noise.
        :param theta: Inertial term for drift..
        :param sigma: Standard deviation of noise.
        """
        self.nA = nA
        self.N = N
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.nA, self.N) * self.mu
        self.reset()

    def reset(self):
        """
        Reset the noise process.
        :return:
        """
        self.state = np.ones(self.nA, self.N) * self.mu

    def noise(self):
        """
        Compute the next noise value.
        :return: The noise value.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(np.shape(x))
        self.state = x + dx
        return self.state * self.scale


class DDPG(object):

    @staticmethod
    def hard_update(target, source):
        """
        Copy parameters from source to target.
        :param target: Target network.
        :param source: Source network.
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def soft_update(target, source, tau):
        """
        Soft update parameters from source to target according to parameter Tau.
        :param target: Target network.
        :param source: Source network.
        :param tau: Weight of the update.
        :return: None
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + tau * param.data)

    def __init__(self, nS, nA, K, device, hidden_size=16, gamma=0.99, tau=0.5):
        """
        Initialize the DDPG networks.
        :param nS: Size of state space.
        :param nA: Size of action space.
        :param hidden_size: Size of hidden layers.
        """
        # Device
        self.device = device

        actor_layers = [nS, hidden_size, hidden_size, nA]
        critic_layers = [nS + nA, hidden_size, hidden_size, 1]

        # Define Networks
        self.actor = Actor(actor_layers, K).to(self.device)
        self.actor_target = Actor(actor_layers, K).to(self.device)
        self.critic = Critic(critic_layers, K).to(self.device)
        self.critic_target = Critic(critic_layers, K).to(self.device)

        # Define Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        # Constants
        self.gamma = gamma
        self.tau = tau

        # Initialize Target Networks
        DDPG.hard_update(self.actor_target, self.actor)
        DDPG.hard_update(self.critic_target, self.critic)

    def select_action(self, state, graph_shift_op, action_noise=None):
        """
        Evaluate the Actor network over the given state, and with injection of noise.
        :param state: The current state.
        :param action_noise: The action noise
        :return:
        """
        self.actor.eval()  # Switch the actor network to Evaluation Mode.
        mu = self.actor(Variable(state), Variable(graph_shift_op)).to(self.device)

        self.actor.train()  # Switch back to Train mode.
        mu = mu.data

        if action_noise is not None:  # Add noise if provided.
            mu += torch.Tensor(action_noise.noise()).to(self.device)

        return mu #mu.clamp(-1, 1) # TODO clamp action to what space?

    def gradient_step(self, batch):
        """
        Take a gradient step given a batch of sampled transitions.
        :param batch: The batch of training samples.
        :return: The loss function in the network.
        """
        #TODO
        # Collect Batch Data
        gso_batch = Variable(torch.cat(batch.gso)).to(self.device)
        next_gso_batch = Variable(torch.cat(batch.next_gso)).to(self.device)

        state_batch = Variable(torch.cat(batch.state)).to(self.device)
        action_batch = Variable(torch.cat(batch.action)).to(self.device)
        reward_batch = Variable(torch.cat(batch.reward)).unsqueeze(1).to(self.device)
        done_batch = Variable(torch.cat(batch.done)).unsqueeze(1).to(self.device)
        next_state_batch = Variable(torch.cat(batch.next_state)).to(self.device)

        # Determine next action and Target Q values
        next_action_batch = self.actor_target(next_state_batch, next_gso_batch)

        next_state_action_values = self.critic_target(next_state_batch, next_action_batch, next_gso_batch)

        expected_state_action_batch = reward_batch + (self.gamma * done_batch * next_state_action_values)

        # Optimize Critic
        self.critic_optim.zero_grad()  # Reset Gradient to Zero
        action_value_batch = self.critic(state_batch, action_batch, gso_batch)  # Evaluate Current Q values for batch

        critic_loss = F.mse_loss(action_value_batch, expected_state_action_batch)
        critic_loss.backward()
        self.critic_optim.step()
        # End Optimize Critic

        # Optimize Actor
        self.actor_optim.zero_grad()
        # Loss related to sampled Actor Gradient.
        policy_loss = -self.critic(state_batch, self.actor(state_batch, gso_batch), gso_batch).mean()
        policy_loss.backward()
        self.actor_optim.step()
        # End Optimize Actor

        # Write parameters to Target networks.
        DDPG.soft_update(self.actor_target, self.actor, self.tau)
        DDPG.soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss.item(), policy_loss.item()

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        """
        Save the Actor and Critic Models after training is completed.
        :param env_name: The environment name.
        :param suffix: The optional suffix.
        :param actor_path: The path to save the actor.
        :param critic_path: The path to save the critic.
        :return: None
        """
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/dppg_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        """
        Load Actor and Critic Models from given paths.
        :param actor_path: The actor path.
        :param critic_path: The critic path.
        :return: None
        """
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path).to(self.device))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path).to(self.device))


def train_ddpg(env, n_agents, device):

    memory = ReplayBuffer(max_size=args.buffer_size)
    ounoise = OUNoise(env.action_space.shape[0], n_agents, scale=1)
    learner = DDPG(env.observation_space.shape[0], env.action_space.shape[0], n_agg, device=device)

    rewards = []
    total_numsteps = 0
    updates = 0

    n_episodes = 250

    for i in range(n_episodes):
        ounoise.reset()
        state = torch.Tensor([env.reset()]).to(device)

        episode_reward = 0
        done = False
        while not done:
            action = learner.select_action(state, ounoise)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            total_numsteps += 1
            episode_reward += reward

            # action = torch.Tensor(action)
            notdone = torch.Tensor([not done]).to(device)
            next_state = torch.Tensor([next_state]).to(device)
            reward = torch.Tensor([reward]).to(device)

            memory.insert(Transition(state, action, notdone, next_state, reward))

            state = next_state

            if memory.curr_size > args.batch_size:
                for _ in range(args.updates_per_step):
                    transitions = memory.sample(args.batch_size)
                    batch = Transition(*zip(*transitions))
                    value_loss, policy_loss = learner.gradient_step(batch)
                    updates += 1

        rewards.append(episode_reward)
        if i % 10 == 0:
            state = torch.Tensor([env.reset()]).to(device)
            episode_reward = 0
            while True:
                action = learner.select_action(state)

                next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                episode_reward += reward
                next_state = torch.Tensor([next_state]).to(device)

                state = next_state
                if done:
                    break

            rewards.append(episode_reward)
            print("Episode: {}, updates: {}, total numsteps: {}, reward: {}, average reward: {}".format(i, updates,
                                                                                                        total_numsteps,
                                                                                                        rewards[-1],
                                                                                                        np.mean(rewards[
                                                                                                                -10:])))

    env.close()
    learner.save_model(args.env)


if __name__ == "__main__":
    env = gym.make(args.env)
    env.seed(1)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_ddpg(env, args.n_agents, device)
