import argparse
import numpy as np
import os
from collections import namedtuple
import random
import gym
import gym_flock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

''' Parse Arguments'''
parser = argparse.ArgumentParser(description='DDPG Implementation')
parser.add_argument('--env', type=str, default="FlockingRelative-v0", help='Gym environment to run')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--buffer_size', type=int, default=10000, help='Replay Buffer Size')
parser.add_argument('--updates_per_step', type=int, default=1, help='Updates per Batch')
parser.add_argument('--n_agents', type=int, default=40, help='n_agents')
parser.add_argument('--n_actions', type=int, default=2, help='n_actions')
parser.add_argument('--n_states', type=int, default=6, help='n_states')
parser.add_argument('--k', type=int, default=2, help='k')
parser.add_argument('--hidden_size', type=int, default=32, help='hidden layer size')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
parser.add_argument('--tau', type=float, default=0.5, help='tau')
parser.add_argument('--seed', type=int, default=9, help='random_seed')
parser.add_argument('--actor_lr', type=float, default=2e-5, help='learning rate for actor')

args = parser.parse_args()

Transition = namedtuple('Transition', ('state', 'action', 'done', 'next_state', 'reward'))

# TODO: how to deal with bounded/unbounded action spaces?? Should I always assume bounded actions?


class ReplayBuffer(object):
    """
    Stores training samples for RL algorithms, represented as tuples of (S, A, R, S).
    """

    def __init__(self, max_size=1000):
        """
        Initialize the replay buffer object. Once the buffer is full, remove the oldest sample.
        :param max_size: maximum size of the buffer.
        """
        self.buffer = []
        self.max_size = max_size
        self.curr_size = 0
        self.position = 0

    def insert(self, sample):
        """
        Insert sample into buffer.
        :param sample: The (S,A,R,S) tuple.
        :return: None
        """
        if self.curr_size < self.max_size:
            self.buffer.append(None)
            self.curr_size = self.curr_size + 1

        self.buffer[self.position] = Transition(*sample)
        self.position = (self.position + 1) % self.max_size

    def sample(self, num_samples):
        """
        Sample a number of transitions from the replay buffer.
        :param num_samples: Number of transitions to sample.
        :return: The set of sampled transitions.
        """
        return random.sample(self.buffer, num_samples)

    def clear(self):
        """
        Clears the current buffer.
        :return: None
        """
        self.buffer = []
        self.curr_size = 0
        self.position = 0


class Actor(nn.Module):

    def __init__(self, n_s, n_a, hidden_layers, k, ind_agg):
        """
        The policy network is allowed to have only one aggregation operation due to communication latency, but we can
        have any number of hidden layers to be executed by each agent individually.
        :param n_s: number of MDP states per agent
        :param n_a: number of MDP actions per agent
        :param hidden_layers: list of ints that will determine the width of each hidden layer
        :param k: aggregation filter length
        :param ind_agg: before which MLP layer index to aggregate
        """
        super(Actor, self).__init__()
        self.k = k
        self.n_s = n_s
        self.n_a = n_a
        self.layers = [n_s] + hidden_layers + [n_a]
        self.n_layers = len(self.layers) - 1

        self.ind_agg = ind_agg  # before which conv layer the aggregation happens

        self.conv_layers = []

        for i in range(0, self.n_layers):

            if i == self.ind_agg:  # after the GSO, reduce dimensions
                step = k
            else:
                step = 1

            m = nn.Conv2d(in_channels=self.layers[i], out_channels=self.layers[i + 1], kernel_size=(step, 1),
                          stride=(step, 1))

            self.conv_layers.append(m)

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

    def forward(self, delay_state, delay_gso):
        """
        The policy relies on delayed information from neighbors. During training, the full history for k time steps is
        necessary.
        :param delay_state: History of states: x_t, x_t-1,...x_t-k+1 of shape (B,K,F,N)
        :param delay_gso: Delayed GSO: [I, A_t, A_t A_t-1, A_t ... A_t-k+1] of shape (B,K,N,N)
        :return:
        """
        batch_size = delay_state.shape[0]
        n_agents = delay_state.shape[3]
        assert delay_gso.shape[0] == batch_size
        assert delay_gso.shape[2] == n_agents
        assert delay_gso.shape[3] == n_agents

        assert delay_state.shape[1] == self.k
        assert delay_state.shape[2] == self.n_s
        assert delay_gso.shape[1] == self.k

        x = delay_state
        x = x.permute(0, 2, 1, 3)  # now (B,F,K,N)

        for i in range(self.n_layers):

            if i == self.ind_agg:  # aggregation only happens once - otherwise each agent operates independently
                x = x.permute(0, 2, 1, 3)  # now (B,K,F,N)
                x = torch.matmul(x, delay_gso)
                x = x.permute(0, 2, 1, 3)  # now (B,F,K,N)

            x = self.conv_layers[i](x)  # now (B,G,1,N)

            if i < self.n_layers - 1:  # last layer - no relu
                #x = self.layer_norms[i](x)
                x = F.relu(x) #torch.tanh(x) #F.relu(x)
            # else:
            #     x = 10 * torch.tanh(x)

        x = x.view((batch_size, 1, self.n_a, n_agents))  # now size (B, 1, nA, N)

        #x = x.clamp(-10, 10)  # TODO these limits depend on the MDP

        return x


class ImitationLearning(object):

    def __init__(self, device, args):  # , n_s, n_a, k, device, hidden_size=32, gamma=0.99, tau=0.5):
        """
        Initialize the DDPG networks.
        :param device: CUDA device for torch
        :param args: experiment arguments
        """

        n_s = args.n_states
        n_a = args.n_actions
        k = args.k
        hidden_size = args.hidden_size
        gamma = args.gamma
        tau = args.tau

        self.n_agents = args.n_agents
        self.n_states = args.n_states
        self.n_actions = args.n_actions

        # Device
        self.device = device

        hidden_layers = [hidden_size, hidden_size]
        ind_agg = 0  # int(len(hidden_layers) / 2)  # aggregate halfway

        # Define Networks
        self.actor = Actor(n_s, n_a, hidden_layers, k, ind_agg).to(self.device)

        # Define Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)

        # Constants
        self.gamma = gamma
        self.tau = tau


    def select_action(self, state):
        """
        Evaluate the Actor network over the given state, and with injection of noise.
        :param state: The current state.
        :param graph_shift_op: History of graph shift operators
        :param action_noise: The action noise
        :return:
        """
        self.actor.eval()  # Switch the actor network to Evaluation Mode.
        mu = self.actor(state.delay_state, state.delay_gso) #.to(self.device)

        # mu is (B, 1, nA, N), need (N, nA)
        mu = mu.permute(0, 1, 3, 2)
        mu = mu.view((self.n_agents, self.n_actions))

        self.actor.train()  # Switch back to Train mode.
        mu = mu.data
        return mu

        #return mu.clamp(-1, 1)  # TODO clamp action to what space?

    def gradient_step(self, batch):
        """
        Take a gradient step given a batch of sampled transitions.
        :param batch: The batch of training samples.
        :return: The loss function in the network.
        """

        delay_gso_batch = Variable(torch.cat(tuple([s.delay_gso for s in batch.state]))).to(self.device)
        delay_state_batch = Variable(torch.cat(tuple([s.delay_state for s in batch.state]))).to(self.device)
        actor_batch = self.actor(delay_state_batch, delay_gso_batch)
        optimal_action_batch = Variable(torch.cat(batch.action)).to(self.device)

        # Optimize Actor
        self.actor_optim.zero_grad()
        # Loss related to sampled Actor Gradient.
        policy_loss = F.mse_loss(actor_batch, optimal_action_batch)
        policy_loss.backward()
        self.actor_optim.step()
        # End Optimize Actor

        return policy_loss.item()

    def save_model(self, env_name, suffix="", actor_path=None):
        """
        Save the Actor Model after training is completed.
        :param env_name: The environment name.
        :param suffix: The optional suffix.
        :param actor_path: The path to save the actor.
        :return: None
        """
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix)
        print('Saving model to {}'.format(actor_path))
        torch.save(self.actor.state_dict(), actor_path)


    def load_model(self, actor_path):
        """
        Load Actor Model from given paths.
        :param actor_path: The actor path.
        :return: None
        """
        print('Loading model from {}'.format(actor_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path).to(self.device))


class MultiAgentStateWithDelay(object):

    def __init__(self, device, args, env_state, prev_state=None):
        """
        Create the state object that keeps track of the current state and GSO and history information
        :param device: CUDA device to use with PyTorch
        :param args:
        :param env_state:
        :param prev_state:
        """
        n_states = args.n_states
        n_agents = args.n_agents
        k = args.k

        # split up the state tuple
        state_value, state_network = env_state

        assert state_value.shape == (n_agents, n_states)
        assert state_network.shape == (n_agents, n_agents)
        assert np.sum(np.diag(state_network)) == 0  # assume no self loops

        # reshape values and network to correct shape
        state_value = state_value.transpose(1, 0)
        state_value = state_value.reshape((1, 1, n_states, n_agents))
        state_network = state_network.reshape((1, 1, n_agents, n_agents))

        # move matrices to GPU
        self.values = torch.Tensor(state_value).to(device)
        self.network = torch.Tensor(state_network).to(device)

        # compute current GSO - powers of the network matrix: current GSO: I, A_t, A_t^2... A_t^k-1
        self.curr_gso = torch.zeros((1, k, n_agents, n_agents)).to(device)
        self.curr_gso[0, 0, :, :] = torch.eye(n_agents).view((1, 1, n_agents, n_agents)).to(device)  # I
        for k_ind in range(1, k):
            self.curr_gso[0, k_ind, :, :] = torch.matmul(self.network, self.curr_gso[0, k_ind - 1, :, :])

        # delayed GSO: I, A_t-1, ...,  A_t-1 * ... * A_t-k
        self.delay_gso = torch.zeros((1, k, n_agents, n_agents)).to(device)
        self.delay_gso[0, 0, :, :] = torch.eye(n_agents).view((1, 1, n_agents, n_agents)).to(device)  # I
        if prev_state is not None and k > 1:
            self.delay_gso[0, 1:k, :, :] = torch.matmul(self.network, prev_state.delay_gso[0, 0:k - 1, :, :])

        # delayed x values x_t, x_t-1,..., x_t-k
        self.delay_state = torch.zeros((1, k, n_states, n_agents)).to(device)
        self.delay_state[0, 0, :, :] = self.values
        if prev_state is not None and k > 1:
            self.delay_state[0, 1:k, :, :] = prev_state.delay_state[0, 0:k - 1, :, :]


def train_ddpg(env, args, device):
    memory = ReplayBuffer(max_size=args.buffer_size)
    learner = ImitationLearning(device, args)

    rewards = []
    total_numsteps = 0
    updates = 0

    n_episodes = 10000

    beta = 1
    beta_coeff = 0.99

    for i in range(n_episodes):

        beta = beta * beta_coeff

        state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)

        episode_reward = 0
        done = False
        while not done:

            optimal_action = env.env.controller()
            if np.random.binomial(1, beta) > 0:
                action = optimal_action
            else:
                action = learner.select_action(state)
                action = action.cpu().numpy()

            next_state, reward, done, _ = env.step(action)

            next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state)

            total_numsteps += 1
            episode_reward += reward

            # action = torch.Tensor(action)
            notdone = torch.Tensor([not done]).to(device)
            reward = torch.Tensor([reward]).to(device)

            # action is (N, nA), need (B, 1, nA, N)
            optimal_action = torch.Tensor(optimal_action).to(device)
            optimal_action = optimal_action.transpose(1, 0)
            optimal_action = optimal_action.reshape((1, 1, args.n_actions, args.n_agents))

            memory.insert(Transition(state, optimal_action, notdone, next_state, reward))

            state = next_state

            if memory.curr_size > args.batch_size:
                for _ in range(args.updates_per_step):
                    transitions = memory.sample(args.batch_size)
                    batch = Transition(*zip(*transitions))
                    policy_loss = learner.gradient_step(batch)
                    updates += 1

        rewards.append(episode_reward)
        # print(i)
        # print(episode_reward)
        if i % 10 == 0:

            episode_reward = 0
            n_eps = 10
            for n in range(n_eps):
                # state = torch.Tensor([env.reset()]).to(device)
                state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)

                done = False
                while not done:
                    action = learner.select_action(state)
                    action = action.cpu().numpy()
                    #action = env.env.controller()
                    next_state, reward, done, _ = env.step(action)
                    next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state)
                    episode_reward += reward
                    state = next_state

            episode_reward = episode_reward / n_eps

            print("Episode: {}, updates: {}, total numsteps: {}, reward: {}, average reward: {}".format(i, updates,
                                                                                                        total_numsteps,
                                                                                                        episode_reward,
                                                                                                        np.mean(rewards[
                                                                                                                -10:])))

    env.close()
    learner.save_model(args.env)




def main():
    # initialize gym env
    env = gym.make(args.env)

    # use seed
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # initialize params tuple
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_ddpg(env, args, device)


if __name__ == "__main__":
    main()
