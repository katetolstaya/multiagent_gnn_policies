import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from learner.state_with_message import MultiAgentState
from learner.replay_buffer import ReplayBuffer
from learner.replay_buffer import Transition
from learner.actor_pac import Actor

# TODO: how to deal with bounded/unbounded action spaces?? Should I always assume bounded actions?


class DAggerPAC(object):

    def __init__(self, device, args, k=None):  # , n_s, n_a, k, device, hidden_size=32, gamma=0.99, tau=0.5):
        """
        Initialize the DDPG networks.
        :param device: CUDA device for torch
        :param args: experiment arguments
        """

        msg_len = args.getint('msg_len')
        n_s = args.getint('n_states')
        n_a = args.getint('n_actions')
        k = k or args.getint('k')
        hidden_size = args.getint('hidden_size')
        n_layers = args.getint('n_layers') or 2
        gamma = args.getfloat('gamma')
        tau = args.getfloat('tau')

        self.grad_clipping = args.getfloat('grad_clipping')

        self.n_agents = args.getint('n_agents')
        self.n_states = n_s
        self.n_actions = n_a
        self.msg_len = msg_len

        # Device
        self.device = device

        hidden_layers = [hidden_size] * n_layers
        # ind_agg = 0  # int(len(hidden_layers) / 2)  # aggregate halfway

        # Define Networks
        self.actor = Actor(n_s, n_a, msg_len, hidden_layers).to(self.device)

        # Define Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=args.getfloat('actor_lr'))

        # Constants
        self.gamma = gamma
        self.tau = tau

    def step(self, state):
        """
        Evaluate the Actor network over the given state
        :param state: The current state.
        :return:
        """
        self.actor.eval()  # Switch the actor network to Evaluation Mode.
        action, message = self.actor(state.value, state.network, state.message)  # .to(self.device)

        # mu is (B, 1, nA, N), need (N, nA)
        action = action.permute(0, 1, 3, 2)
        action = action.view((self.n_agents, self.n_actions))

        message = message.permute(0, 1, 3, 2)
        message = message.view((self.n_agents, self.msg_len))

        self.actor.train()  # Switch back to Train mode.
        action = action.data
        message = message.data
        return action, message

        # return mu.clamp(-1, 1)  # TODO clamp action to what space?

    def gradient_step(self, batch):
        """
        Take a gradient step given a batch of sampled transitions.
        :param batch: The batch of training samples, with each sample of length unroll transitions
        :return: The loss function in the network.
        """

        value_batch = []
        network_batch = []
        message_batch = []
        optimal_action_batch = []

        for transitions in batch:
            value_batch.append(torch.cat(tuple([t.state.value for t in transitions]), 1))
            network_batch.append(torch.cat(tuple([t.state.network for t in transitions]), 1))
            message_batch.append(torch.cat(tuple([t.state.message for t in transitions]), 1))
            optimal_action_batch.append(torch.cat(tuple([t.action for t in transitions]), 1))
            # optimal_action_batch.append(transitions[-1].action)

        value_batch = Variable(torch.cat(tuple(value_batch))).to(self.device)
        network_batch = Variable(torch.cat(tuple(network_batch))).to(self.device)
        message_batch = Variable(torch.cat(tuple(message_batch))).to(self.device)
        optimal_action_batch = Variable(torch.cat(tuple(optimal_action_batch))).to(self.device)
        actor_batch, _ = self.actor(value_batch, network_batch, message_batch)

        self.actor_optim.zero_grad()
        # Loss related to sampled Actor Gradient.
        policy_loss = F.mse_loss(actor_batch[:, -1, :, :], optimal_action_batch[:, -1, :, :])
        policy_loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.grad_clipping)
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
            actor_path = "models/actor_{}_{}".format(env_name, suffix)
        print('Saving model to {}'.format(actor_path))
        torch.save(self.actor.state_dict(), actor_path)

    def load_model(self, actor_path, map_location):
        """
        Load Actor Model from given paths.
        :param actor_path: The actor path.
        :return: None
        """
        # print('Loading model from {}'.format(actor_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location))
            self.actor.to(self.device)


def train_dagger(env, args, device):
    debug = args.getboolean('debug')
    memory = ReplayBuffer(max_size=args.getint('buffer_size'))
    learner = DAggerPAC(device, args)

    n_a = args.getint('n_actions')
    n_agents = args.getint('n_agents')
    msg_len = args.getint('msg_len')
    batch_size = args.getint('batch_size')
    unroll_length = args.getint('unroll_len')

    n_train_episodes = args.getint('n_train_episodes')

    beta_coeff = args.getfloat('beta_coeff')
    beta_min = args.getfloat('beta_min')
    test_interval = args.getint('test_interval')
    n_test_episodes = args.getint('n_test_episodes')

    total_numsteps = 0
    updates = 0
    beta = 1

    stats = {'mean': -1.0 * np.Inf, 'std': 0}

    for i in range(n_train_episodes):
        beta = max(beta * beta_coeff, beta_min)

        message = np.zeros((n_agents, msg_len))

        unroll = []
        state = MultiAgentState(device, args, env.reset(), message)

        done = False
        policy_loss_sum = 0
        while not done:
            optimal_action = env.env.controller()

            action, message = learner.step(state)
            action = action.cpu().numpy()
            message = message.cpu().numpy()

            if np.random.binomial(1, beta) > 0:
                action = optimal_action

            next_state, reward, done, _ = env.step(action)
            next_state = MultiAgentState(device, args, next_state, message)

            total_numsteps += 1

            # action = torch.Tensor(action)
            notdone = torch.Tensor([not done]).to(device)
            reward = torch.Tensor([reward]).to(device)

            # action is (N, nA), need (B, 1, nA, N)
            optimal_action = torch.Tensor(optimal_action).to(device)
            optimal_action = optimal_action.transpose(1, 0)
            optimal_action = optimal_action.reshape((1, 1, n_a, n_agents))

            transition = Transition(state, optimal_action, notdone, next_state, reward)
            unroll.append(transition)

            if len(unroll) == unroll_length:
                memory.insert(unroll)
                unroll = []
            state = next_state

        if memory.curr_size > batch_size:
            for _ in range(args.getint('updates_per_step')):
                unrolled_transitions = memory.sample(batch_size)
                policy_loss = learner.gradient_step(unrolled_transitions)
                policy_loss_sum += policy_loss
                updates += 1

        if i % test_interval == 0 and debug:
            test_rewards = []
            for _ in range(n_test_episodes):
                ep_reward = 0
                message = np.zeros((n_agents, msg_len))
                state = MultiAgentState(device, args, env.reset(), message)
                done = False
                while not done:
                    action, message = learner.step(state)
                    action = action.cpu().numpy()
                    message = message.cpu().numpy()
                    next_state, reward, done, _ = env.step(action)
                    next_state = MultiAgentState(device, args, next_state, message)
                    ep_reward += reward
                    state = next_state
                    # env.render()
                test_rewards.append(ep_reward)

            mean_reward = np.mean(test_rewards)
            # if stats['mean'] < mean_reward:
            #     stats['mean'] = mean_reward
            #     stats['std'] = np.std(test_rewards)
            #
            #     if debug and args.get('fname'):  # save the best model
            #         learner.save_model(args.get('env'), suffix=args.get('fname'))

            if debug:
                print(
                    "Episode: {}, updates: {}, total numsteps: {}, reward: {}, policy loss: {}".format(
                        i, updates,
                        total_numsteps,
                        mean_reward,
                        policy_loss_sum))

    test_rewards = []
    for _ in range(n_test_episodes):
        ep_reward = 0
        message = np.zeros((n_agents, msg_len))
        state = MultiAgentState(device, args, env.reset(), message)
        done = False
        while not done:
            action, message = learner.step(state)
            action = action.cpu().numpy()
            message = message.cpu().numpy()
            next_state, reward, done, _ = env.step(action)
            next_state = MultiAgentState(device, args, next_state, message)
            ep_reward += reward
            state = next_state
            # env.render()
        test_rewards.append(ep_reward)

    mean_reward = np.mean(test_rewards)
    stats['mean'] = mean_reward
    stats['std'] = np.std(test_rewards)

    if debug and args.get('fname'):  # save the best model
        learner.save_model(args.get('env'), suffix=args.get('fname'))

    env.close()
    return stats
