import argparse
import numpy as np
import os
import random
import gym

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

from learner.state_with_delay import MultiAgentStateWithDelay
from learner.replay_buffer import ReplayBuffer
from learner.replay_buffer import Transition
from learner.actor import Actor
from learner.critic import Critic

''' Parse Arguments'''
parser = argparse.ArgumentParser(description='DDPG Implementation')
parser.add_argument('--env', type=str, default="FlockingRelative-v0", help='Gym environment to run')
parser.add_argument('--n_agents', type=int, default=40, help='n_agents')
parser.add_argument('--n_actions', type=int, default=2, help='n_actions')
parser.add_argument('--n_states', type=int, default=4, help='n_states')
parser.add_argument('--k', type=int, default=1, help='k')
parser.add_argument('--hidden_size', type=int, default=16, help='hidden_size')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
parser.add_argument('--tau', type=float, default=0.5, help='tau')
parser.add_argument('--seed', type=int, default=8, help='random_seed')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--buffer_size', type=int, default=10000, help='Replay Buffer Size')
parser.add_argument('--updates_per_step', type=int, default=1, help='Updates per Batch')

args = parser.parse_args()


class OUNoise:
    """
    Generates noise from an Ornstein Uhlenbeck process, for temporally correlated exploration. Useful for physical
    control problems with inertia. See https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(self, n_a, n_agents, scale=0.05, mu=0, theta=0.15, sigma=0.2):  # TODO change default params
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
        self.state = np.ones((self.n_agents, self.nA)) * self.mu
        self.reset()

    def reset(self):
        """
        Reset the noise process.
        :return:
        """
        self.state = np.ones((self.n_agents, self.nA)) * self.mu

    def noise(self):
        """
        Compute the next noise value.
        :return: The noise value.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(np.shape(x))
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
        ind_agg = int(len(hidden_layers) / 2)  # aggregate halfway

        # Define Networks
        self.actor = Actor(n_s, n_a, hidden_layers, k, ind_agg).to(self.device)
        self.actor_target = Actor(n_s, n_a, hidden_layers, k, ind_agg).to(self.device)
        self.critic = Critic(n_s, n_a, hidden_layers, k).to(self.device)
        self.critic_target = Critic(n_s, n_a, hidden_layers, k).to(self.device)

        # Define Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-4)

        # Constants
        self.gamma = gamma
        self.tau = tau

        # Initialize Target Networks
        DDPG.hard_update(self.actor_target, self.actor)
        DDPG.hard_update(self.critic_target, self.critic)

    def select_action(self, state, action_noise=None):
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

        if action_noise is not None:  # Add noise if provided.
            mu += torch.Tensor(action_noise.noise()).to(self.device)

        return mu.clamp(-1, 1)  # TODO clamp action to what space?

    def gradient_step(self, batch):
        """
        Take a gradient step given a batch of sampled transitions.
        :param batch: The batch of training samples.
        :return: The loss function in the network.
        """

        # Collect Batch Data
        # use MultiAgentStateWithDelay object to prepare data:
        # for Critic - current centralized data
        # for Actor - delayed decentralized data

        delay_gso_batch = Variable(torch.cat(tuple([s.delay_gso for s in batch.state]))).to(self.device)
        gso_batch = Variable(torch.cat(tuple([s.curr_gso for s in batch.state]))).to(self.device)
        next_delay_gso_batch = Variable(torch.cat(tuple([s.delay_gso for s in batch.next_state]))).to(self.device)
        next_gso_batch = Variable(torch.cat(tuple([s.curr_gso for s in batch.next_state]))).to(self.device)

        delay_state_batch = Variable(torch.cat(tuple([s.delay_state for s in batch.state]))).to(self.device)
        state_batch = Variable(torch.cat(tuple([s.values for s in batch.state]))).to(self.device)
        next_delay_state_batch = Variable(torch.cat(tuple([s.delay_state for s in batch.next_state]))).to(self.device)
        next_state_batch = Variable(torch.cat(tuple([s.values for s in batch.next_state]))).to(self.device)

        action_batch = Variable(torch.cat(batch.action)).to(self.device)
        reward_batch = Variable(torch.cat(batch.reward)).unsqueeze(1).to(self.device)
        done_batch = Variable(torch.cat(batch.done)).unsqueeze(1).to(self.device)

        # Determine next action and Target Q values
        next_action_batch = self.actor_target(next_delay_state_batch, next_delay_gso_batch)
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
        policy_loss = -self.critic(state_batch, self.actor(delay_state_batch, delay_gso_batch), gso_batch).mean()
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
    ounoise = OUNoise(args.n_actions, args.n_agents, scale=1)
    learner = DDPG(device, args)

    rewards = []
    total_numsteps = 0
    updates = 0

    n_episodes = 10000

    for i in range(n_episodes):
        ounoise.reset()

        state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)

        episode_reward = 0
        done = False
        while not done:
            action = learner.select_action(state, ounoise)
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state)

            total_numsteps += 1
            episode_reward += reward

            # action = torch.Tensor(action)
            notdone = torch.Tensor([not done]).to(device)
            reward = torch.Tensor([reward]).to(device)

            # action is (N, nA), need (B, 1, nA, N)
            action = action.transpose(1, 0)
            action = action.reshape((1, 1, args.n_actions, args.n_agents))

            memory.insert(Transition(state, action, notdone, next_state, reward))

            state = next_state

            if memory.curr_size > args.batch_size:
                for _ in range(args.updates_per_step):
                    transitions = memory.sample(args.batch_size)
                    batch = Transition(*zip(*transitions))
                    value_loss, policy_loss = learner.gradient_step(batch)
                    updates += 1

        rewards.append(episode_reward)
        # print(i)
        # print(episode_reward)
        if i % 10 == 0:
            # state = torch.Tensor([env.reset()]).to(device)
            state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)
            episode_reward = 0
            done = False
            while not done:
                action = learner.select_action(state)
                next_state, reward, done, _ = env.step(action.cpu().numpy())
                next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state)
                episode_reward += reward
                state = next_state
                #env.render()

            rewards.append(episode_reward)
            print("Episode: {}, updates: {}, total numsteps: {}, reward: {}, average reward: {}".format(i, updates,
                                                                                                        total_numsteps,
                                                                                                        rewards[-1],
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
