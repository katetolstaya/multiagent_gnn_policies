import argparse
import numpy as np
import os

import random
import gym
import gym_flock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

from state_with_delay import MultiAgentStateWithDelay
from replay_buffer import ReplayBuffer
from replay_buffer import Transition

''' Parse Arguments'''
parser = argparse.ArgumentParser(description='DDPG Implementation')
parser.add_argument('--env', type=str, default="FlockingRelative-v0", help='Gym environment to run')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
parser.add_argument('--buffer_size', type=int, default=5000, help='Replay Buffer Size')
parser.add_argument('--updates_per_step', type=int, default=200, help='Updates per Batch')
parser.add_argument('--n_agents', type=int, default=80, help='n_agents')
parser.add_argument('--n_actions', type=int, default=2, help='n_actions')
parser.add_argument('--n_states', type=int, default=6, help='n_states')
parser.add_argument('--k', type=int, default=2, help='k')
parser.add_argument('--hidden_size', type=int, default=32, help='hidden layer size')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
parser.add_argument('--tau', type=float, default=0.5, help='tau')
parser.add_argument('--seed', type=int, default=11, help='random_seed')
parser.add_argument('--actor_lr', type=float, default=5e-5, help='learning rate for actor')

args = parser.parse_args()




# TODO: how to deal with bounded/unbounded action spaces?? Should I always assume bounded actions?





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
                # x = self.layer_norms[i](x)
                x = torch.tanh(x)
                #x = F.relu(x)  # torch.tanh(x) #F.relu(x)
            # else:
            #     x = 10 * torch.tanh(x)

        x = x.view((batch_size, 1, self.n_a, n_agents))  # now size (B, 1, nA, N)

        # x = x.clamp(-10, 10)  # TODO these limits depend on the MDP

        return x


class DAGGER(object):

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

        hidden_layers = [ hidden_size, hidden_size]
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
        mu = self.actor(state.delay_state, state.delay_gso)  # .to(self.device)

        # mu is (B, 1, nA, N), need (N, nA)
        mu = mu.permute(0, 1, 3, 2)
        mu = mu.view((self.n_agents, self.n_actions))

        self.actor.train()  # Switch back to Train mode.
        mu = mu.data
        return mu

        # return mu.clamp(-1, 1)  # TODO clamp action to what space?

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




def train_ddpg(env, args, device, debug=True):
    memory = ReplayBuffer(max_size=args.buffer_size)
    learner = DAGGER(device, args)

    rewards = []
    total_numsteps = 0
    updates = 0

    n_episodes = 10000

    beta = 1
    beta_coeff = 0.993

    for i in range(n_episodes):

        beta = max(beta * beta_coeff, 0.5)

        state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)

        episode_reward = 0
        done = False
        policy_loss_sum = 0
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
                policy_loss_sum += policy_loss
                updates += 1

        if i % 10 == 0:

            episode_reward = 0
            n_eps = 1
            for n in range(n_eps):
                state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)
                done = False
                while not done:
                    action = learner.select_action(state)
                    next_state, reward, done, _ = env.step(action.cpu().numpy())
                    next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state)
                    episode_reward += reward
                    state = next_state
                    env.render()
            rewards.append(episode_reward)

            if debug:
                print(
                    "Episode: {}, updates: {}, total numsteps: {}, reward: {}, average reward: {}, policy loss: {}".format(
                        i, updates,
                        total_numsteps,
                        rewards[-1],
                        np.mean(rewards[-10:]), policy_loss_sum))

    env.close()
    learner.save_model(args.env)
    return np.mean(rewards[-20:])


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
