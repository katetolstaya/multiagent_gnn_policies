import argparse
import numpy as np
import os
from collections import namedtuple
import random
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable


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


class Critic(nn.Module):

    def __init__(self, n_s, n_a, hidden_layers, k):
        """
        The actor network is centralized, so it can have any number of [GSO -> Linearity -> Activation] layers
        If there's a lot of layers here, it doesn't matter if we have a non-linearity or GSO first (test this).
        :param n_s: number of MDP states per agent
        :param n_a: number of MDP actions per agent
        :param hidden_layers: list of ints that will determine the width of each hidden layer
        :param k: aggregation filter length
        """
        super(Critic, self).__init__()

        self.k = k
        self.n_s = n_s
        self.n_a = n_a

        self.layers = [self.n_s + self.n_a] + hidden_layers + [1]
        self.n_layers = len(self.layers) - 1
        self.conv_layers = []
        self.gso_first = True

        for i in range(self.n_layers):

            if i > 0 or self.gso_first:
                in_channels = k
            else:
                in_channels = 1

            m = nn.Conv2d(in_channels=in_channels, out_channels=self.layers[i + 1], kernel_size=(self.layers[i], 1),
                          stride=(self.layers[i], 1))
            self.conv_layers.append(m)
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

    def forward(self, states, actions, graph_shift_ops):
        """
        Evaluate the critic network
        :param states: Current states of shape (B,1,n_s,N), where B = # batches, n_s = # features, N = # agents
        :param actions: Current actions of shape (B,1,n_a,N), where B = # batches, n_a = # features, N = # agents
        :param graph_shift_ops: Current GSO = [I, A_t, A_t^2,..., A_t^K-1] of shape (B,K,N,N)
        :return:
        """

        batch_size = np.shape(states)[0]
        n_agents = states.shape[3]

        assert batch_size == actions.shape[0]
        assert batch_size == graph_shift_ops.shape[0]
        assert actions.shape[3] == n_agents
        assert states.shape[1] == 1
        assert actions.shape[1] == 1
        assert states.shape[2] == self.n_s
        assert actions.shape[2] == self.n_a
        assert graph_shift_ops.shape[1] == self.k
        assert graph_shift_ops.shape[2] == n_agents
        assert graph_shift_ops.shape[3] == n_agents

        x = torch.cat((states, actions), 2)

        # GSO -> Linearity -> Activation
        for i in range(self.n_layers):

            if i > 0 or self.gso_first:
                x = torch.matmul(x, graph_shift_ops)  # GSO

            x = self.conv_layers[i](x)  # Linear layer

            if i < self.n_layers - 1:  # last layer needs no relu()
                x = F.relu(x)

            x = x.view((batch_size, 1, self.layers[i + 1], n_agents))

        x = x.view((batch_size, 1, n_agents))  # now size (B, 1, N)

        return x  # size of x here is batch_size x output_layer x 1 x n_agents


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

        self.layers = [n_s] + hidden_layers + [n_a]
        self.n_layers = len(self.layers) - 1

        self.ind_agg = ind_agg  # before which conv layer the aggregation happens

        self.conv_layers = []

        for i in range(0, self.n_layers):

            if i == self.ind_agg:  # after the GSO, reduce dimensions
                step = k
            else:
                step = 1

            m = nn.Conv2d(in_channels=self.layers[i], out_channels=self.layers[i + 1], kernel_size=(step, 1), stride=(step, 1))

            self.conv_layers.append(m)

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

    def forward(self, states, graph_shift_ops):
        """
        The policy relies on delayed information from neighbors. During training, the full history for k time steps is
        necessary.
        :param states: History of states: x_t, x_t-1,...x_t-k+1 of shape (B,K,F,N)
        :param graph_shift_ops: Delayed GSO: [I, A_t, A_t A_t-1, A_t ... A_t-k+1] of shape (B,K,N,N)
        :return:
        """
        batch_size = states.shape[0]
        n_agents = states.shape[3]
        assert graph_shift_ops.shape[0] == batch_size
        assert graph_shift_ops.shape[2] == n_agents
        assert graph_shift_ops.shape[3] == n_agents
        assert states.shape[1] == 1
        assert states.shape[2] == self.n_s
        assert graph_shift_ops.shape[1] == self.k

        x = states
        x = x.permute(0, 2, 1, 3)  # now (B,F,K,N)

        for i in range(self.n_layers):

            if i == self.ind_agg:  # aggregation only happens once - otherwise each agent operates independently
                x = x.permute(0, 2, 1, 3)   # now (B,K,F,N)
                x = torch.matmul(x, graph_shift_ops)
                x = x.permute(0, 2, 1, 3)  # now (B,F,K,N)

            x = self.conv_layers[i](x)  # now (B,G,1,N)

            if i < self.n_layers - 1:  # last layer - no relu
                x = F.relu(x)

        x = x.view((batch_size, self.n_a, n_agents))  # now size (B, nA, N)

        return x


class OUNoise:
    """
    Generates noise from an Ornstein Uhlenbeck process, for temporally correlated exploration. Useful for physical
    control problems with inertia. See https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(self, n_a, n_agents, scale=0.3, mu=0, theta=0.15, sigma=0.2):  # TODO change default params
        """
        Initialize the Noise parameters.
        :param n_a: Size of the Action space.
        :param n_agents: Number of agents.
        :param scale: Scale of the noise process.
        :param mu: The mean of the noise.
        :param theta: Inertial term for drift..
        :param sigma: Standard deviation of noise.
        """
        self.nA = n_a
        self.n_agents = n_agents
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.nA, self.n_agents) * self.mu
        self.reset()

    def reset(self):
        """
        Reset the noise process.
        :return:
        """
        self.state = np.ones(self.nA, self.n_agents) * self.mu

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

    def __init__(self, n_s, n_a, k, device, hidden_size=32, gamma=0.99, tau=0.5):
        """
        Initialize the DDPG networks.
        :param n_s: Size of state space.
        :param n_a: Size of action space.
        :param hidden_size: Size of hidden layers.
        """
        # Device
        self.device = device

        hidden_layers = [hidden_size, hidden_size, hidden_size, hidden_size]
        ind_agg = int(len(hidden_layers)/2)  # aggregate halfway

        # Define Networks
        self.actor = Actor(n_s, n_a, hidden_layers, k, ind_agg).to(self.device)
        self.actor_target = Actor(n_s, n_a, hidden_layers, k, ind_agg).to(self.device)
        self.critic = Critic(n_s, n_a, hidden_layers, k).to(self.device)
        self.critic_target = Critic(n_s, n_a, hidden_layers, k).to(self.device)

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
        :param graph_shift_op: History of graph shift operators
        :param action_noise: The action noise
        :return:
        """
        self.actor.eval()  # Switch the actor network to Evaluation Mode.
        mu = self.actor(Variable(state), Variable(graph_shift_op)).to(self.device)

        self.actor.train()  # Switch back to Train mode.
        mu = mu.data

        if action_noise is not None:  # Add noise if provided.
            mu += torch.Tensor(action_noise.noise()).to(self.device)

        return mu  # mu.clamp(-1, 1) # TODO clamp action to what space?

    def gradient_step(self, batch):
        """
        Take a gradient step given a batch of sampled transitions.
        :param batch: The batch of training samples.
        :return: The loss function in the network.
        """
        # TODO
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


class MultiAgentStateWithDelay(object):


    # TODO n_s, n_a, n_agents, k, are shared parameters
    # xt, gso, prev_state not shared
    def __init__(self, n_s, n_a, n_agents, k, st, gso, prev_state=None):
        """
        Do not need the delayed actions. This object only stores state & GSO information
        :param n_s:
        :param n_a:
        :param n_agents:
        :param k:
        :param st:
        :param gso:
        :param prev_state:
        """
        # 0) just the current state value: x_t
        # 1) delayed x values x_t, x_t-1,..., x_t-k
        # 2) current GSO: I, A_t, A_t^2... A_t^k
        # 3) delayed GSO: I, A_t-1, ...,  A_t-1 * ... * A_t-k

        f = 0 # TODO num features - get from dimension of state

        self.st = st
        self.curr_gso = np.zeros((1, k, n_agents, n_agents))

        if prev_state is None:
            self.delayed_x = np.zeros((1, k, f, n_agents))
            self.delayed_gso = np.zeros((1, k, n_agents, n_agents))
        else:
            self.delayed_gso = np.copy(prev_state.delayed_gso)
            # TODO: multiply by curr network, shift, fill I

            self.delayed_s = np.copy(prev_state.delayed_s)
            # TODO: shift, fill with current state



# TODO this function and then adjust the env to return state = (values, network)
def train_ddpg(env, n_agents, device):

    memory = ReplayBuffer(max_size=args.buffer_size)
    ounoise = OUNoise(env.action_space.shape[0], n_agents, scale=1)
    learner = DDPG(env.observation_space.shape[0], env.action_space.shape[0], K, device=device)

    rewards = []
    total_numsteps = 0
    updates = 0

    n_episodes = 250

    for i in range(n_episodes):
        ounoise.reset()

        env_state = env.reset()  # a tuple (values, network)
        state = torch.Tensor([env_state[0]]).to(device)
        network = torch.Tensor([env_state[1]])

        # TODO remove network self loops and normalize

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

            # memory needs to store:
            # 0) just the current x value: x_t
            # 1) delayed x values x_t, x_t-1,..., x_t-k
            # 2) current GSO: I, A_t, A_t^2... A_t^k
            # 3) delayed GSO: I, A_t-1, ...,  A_t-1 * ... * A_t-k

            # OK, but what to do about the next state? Maybe make a named tuple for state that has all of these friends
            # in it, so that I don't have to duplicate data... or will it be duplicated regardless?
            # Maybe a MultiAgentStateWithDelay object to store these wonderful matrices?

            # for delayed info, https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html
            # but how to do this in-place efficiently?

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
