import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from learner.state_with_message import MultiAgentState
from learner.replay_buffer import ReplayBuffer
from learner.replay_buffer import Transition
from learner.message_pac import Actor, Critic, Message


class PACDDPG(object):

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

    def __init__(self, device, args):
        """
        Initialize the DDPG networks.
        :param device: CUDA device for torch
        :param args: experiment arguments
        """

        msg_len = args.getint('msg_len')
        n_s = args.getint('n_states')
        n_a = args.getint('n_actions')
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

        # Define Networks
        self.actor = Actor(n_s, n_a, msg_len, hidden_layers).to(self.device)
        self.actor_target = Actor(n_s, n_a, msg_len, hidden_layers).to(self.device)

        self.critic = Critic(n_s, n_a, msg_len, hidden_layers).to(self.device)
        self.critic_target = Critic(n_s, n_a, msg_len, hidden_layers).to(self.device)

        self.message = Message(n_s, msg_len, hidden_layers).to(self.device)
        self.message_target = Message(n_s, msg_len, hidden_layers).to(self.device)

        # Define Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=args.getfloat('actor_lr'))
        self.critic_optim = Adam(self.critic.parameters(), lr=args.getfloat('critic_lr'))
        self.message_optim = Adam(self.message.parameters(), lr=args.getfloat('message_lr'))

        # Initialize Target Networks
        PACDDPG.hard_update(self.actor_target, self.actor)
        PACDDPG.hard_update(self.critic_target, self.critic)
        PACDDPG.hard_update(self.message_target, self.message)

        # Constants
        self.gamma = gamma
        self.tau = tau

    def step(self, state, action_noise=None):
        """
        Evaluate the Actor network over the given state
        :param state: The current state.
        :param action_noise:
        :return:
        """
        self.actor.eval()  # Switch the actor network to Evaluation Mode.
        self.message.eval()  # Switch the actor network to Evaluation Mode.

        msg_val = self.message(state.value, state.network, state.message)
        action = self.actor(msg_val)  # .to(self.device)

        # mu is (B, 1, nA, N), need (N, nA)
        action = action.permute(0, 1, 3, 2)
        action = action.view((self.n_agents, self.n_actions))

        if self.msg_len > 0:
            message = msg_val[:, :, -self.msg_len:, :]
            message = message.permute(0, 1, 3, 2)
            message = message.view((self.n_agents, self.msg_len))

        self.actor.train()  # Switch back to Train mode.
        self.message.train()  # Switch back to Train mode.

        action = action.data

        if self.msg_len > 0:
            message = message.data
        else:
            message = None

        if action_noise is not None:  # Add noise if provided.
            action += torch.Tensor(action_noise.noise()).to(self.device)
            action.clamp(-1, 1)

        return action, message

    def gradient_step(self, batch):
        """
        Take a gradient step given a batch of sampled transitions.
        :param batch: The batch of training samples, with each sample of length unroll transitions
        :return: The loss function in the network.
        """

        ################################################################################################################
        # Evaluate message passing networks using T-step unrolled transitions
        value_batch = []
        network_batch = []
        message_batch = []

        for transitions in batch:
            value_batch.append(torch.cat(tuple([t.state.value for t in transitions]), 1))
            network_batch.append(torch.cat(tuple([t.state.network for t in transitions]), 1))
            if self.msg_len > 0:
                message_batch.append(torch.cat(tuple([t.state.message for t in transitions]), 1))
            else:
                message_batch = None

        value_batch = Variable(torch.cat(tuple(value_batch))).to(self.device)
        network_batch = Variable(torch.cat(tuple(network_batch))).to(self.device)
        if self.msg_len > 0:
            message_batch = Variable(torch.cat(tuple(message_batch))).to(self.device)
        else:
            message_batch = None

        state_msg_batch = self.message(value_batch, network_batch, message_batch)

        # TODO - should we unroll the target? Or just keep the latest message? improved stability?
        # next_value_batch = []
        # next_network_batch = []
        # next_message_batch = []
        #
        # for transitions in batch:
        #     next_value_batch.append(torch.cat(tuple([t.next_state.value for t in transitions]), 1))
        #     next_network_batch.append(torch.cat(tuple([t.next_state.network for t in transitions]), 1))
        #     next_message_batch.append(torch.cat(tuple([t.next_state.message for t in transitions]), 1))

        next_value_batch = [transitions[-1].next_state.value for transitions in batch]
        next_network_batch = [transitions[-1].next_state.network for transitions in batch]
        if self.msg_len > 0:
            next_message_batch = [transitions[-1].next_state.message for transitions in batch]
        else:
            next_message_batch = None

        next_value_batch = Variable(torch.cat(tuple(next_value_batch))).to(self.device)
        next_network_batch = Variable(torch.cat(tuple(next_network_batch))).to(self.device)

        if self.msg_len > 0:
            next_message_batch = Variable(torch.cat(tuple(next_message_batch))).to(self.device)
        else:
            next_message_batch = None

        next_state_msg_batch = self.message_target(next_value_batch, next_network_batch, next_message_batch)

        ################################################################################################################
        # Get single step batch - use last message, state, action for evaluating the critic and actor
        action_batch = [transitions[-1].action for transitions in batch]
        reward_batch = [transitions[-1].reward for transitions in batch]
        mask_batch = [transitions[-1].notdone for transitions in batch]

        action_batch = Variable(torch.cat(tuple(action_batch))).to(self.device)
        reward_batch = Variable(torch.cat(tuple(reward_batch))).to(self.device)
        mask_batch = Variable(torch.cat(tuple(mask_batch))).to(self.device)

        reward_batch = reward_batch.view(-1, 1, 1, self.n_agents)
        mask_batch = mask_batch.view(-1, 1, 1, 1)

        ################################################################################################################
        # Evaluate target policy and critic
        next_action_batch = self.actor_target(next_state_msg_batch)
        next_action_value_batch = self.critic_target(next_state_msg_batch, next_action_batch)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_action_value_batch)

        ################################################################################################################
        # Optimize Critic and Message networks
        self.critic_optim.zero_grad()  # Reset Gradient to Zero
        # self.message_optim.zero_grad()
        critic_loss = F.mse_loss(self.critic(state_msg_batch, action_batch), expected_state_action_batch)
        critic_loss.backward(retain_graph=True)

        # TODO - gradient clipping only for message network, not actor or policy?
        self.critic_optim.step()
        # torch.nn.utils.clip_grad_value_(self.message.parameters(), self.grad_clipping)
        # self.message_optim.step()

        ################################################################################################################
        # Optimize Actor network
        # TODO optimize message passing here or no?

        self.actor_optim.zero_grad()
        self.message_optim.zero_grad()
        policy_loss = -self.critic(state_msg_batch, self.actor(state_msg_batch)).mean()
        policy_loss.backward()
        self.actor_optim.step()

        torch.nn.utils.clip_grad_value_(self.message.parameters(), self.grad_clipping)
        self.message_optim.step()

        ################################################################################################################
        # Write parameters to Target networks.
        PACDDPG.soft_update(self.actor_target, self.actor, self.tau)
        PACDDPG.soft_update(self.critic_target, self.critic, self.tau)
        PACDDPG.soft_update(self.message_target, self.message, self.tau)

        return policy_loss.item(), critic_loss.item()

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None, msg_path=None):
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
        if msg_path is None:
            msg_path = "models/dppg_message_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {} and {}'.format(actor_path, critic_path, msg_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.message.state_dict(), msg_path)

    def load_model(self, actor_path, critic_path, msg_path):
        """
        Load Actor and Critic Models from given paths.
        :param actor_path: The actor path.
        :param critic_path: The critic path.
        :return: None
        """
        print('Loading models from {} and {} and {}'.format(actor_path, critic_path, msg_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path).to(self.device))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path).to(self.device))
        if msg_path is not None:
            self.message.load_state_dict(torch.load(msg_path).to(self.device))


def train(env, args, device):
    debug = args.getboolean('debug')
    memory = ReplayBuffer(max_size=args.getint('buffer_size'))
    learner = PACDDPG(device, args)

    n_a = args.getint('n_actions')
    n_agents = args.getint('n_agents')
    msg_len = args.getint('msg_len')
    batch_size = args.getint('batch_size')
    unroll_length = args.getint('unroll_len')

    n_train_episodes = args.getint('n_train_episodes')

    test_interval = args.getint('test_interval')
    n_test_episodes = args.getint('n_test_episodes')

    total_numsteps = 0
    updates = 0

    stats = {'mean': -1.0 * np.Inf, 'std': 0}

    for i in range(n_train_episodes):
        message = np.zeros((n_agents, msg_len))

        unroll = []

        state = MultiAgentState(device, args, env.reset(), message)

        done = False
        policy_loss_sum = 0
        critic_loss_sum = 0
        while not done:

            action, message = learner.step(state)
            action = action.cpu().numpy()
            if msg_len > 0:
                message = message.cpu().numpy()

            next_state, reward, done, _ = env.step(action)


            next_state = MultiAgentState(device, args, next_state, message)

            total_numsteps += 1

            # action = torch.Tensor(action)
            notdone = torch.Tensor([not done]).to(device)
            reward = torch.Tensor([reward]).to(device)

            # action is (N, nA), need (B, 1, nA, N)
            action = torch.Tensor(action).to(device)
            action = action.transpose(1, 0)
            action = action.reshape((1, 1, n_a, n_agents))

            transition = Transition(state, action, notdone, next_state, reward)
            unroll.append(transition)

            if len(unroll) == unroll_length:
                memory.insert(unroll)
                unroll = []
            state = next_state

        if memory.curr_size > batch_size:
            for _ in range(args.getint('updates_per_step')):
                unrolled_transitions = memory.sample(batch_size)
                policy_loss, critic_loss = learner.gradient_step(unrolled_transitions)
                policy_loss_sum += policy_loss
                critic_loss_sum += critic_loss
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
                    if msg_len > 0:
                        message = message.cpu().numpy()
                    else:
                        message = None
                        
                    next_state, reward, done, _ = env.step(action)
                    next_state = MultiAgentState(device, args, next_state, message)
                    ep_reward += np.sum(reward)
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
                    "Episode: {}, updates: {}, total numsteps: {}, reward: {}, policy loss: {}, critic loss: {}".format(
                        i, updates,
                        total_numsteps,
                        mean_reward,
                        policy_loss_sum,
                        critic_loss_sum))

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
