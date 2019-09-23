import torch
import numpy as np


class MultiAgentState(object):

    def __init__(self, device, args, env_state, messages, k=None):
        """
        Create the state object that keeps track of the current state and GSO and history information
        :param device: CUDA device to use with PyTorch
        :param args:
        :param env_state:
        :param prev_state:
        """
        n_states = args.getint('n_states')
        n_agents = args.getint('n_agents')
        msg_len = args.getint('msg_len')
        k = k or args.getint('k')
        # n_states = args.n_states
        # n_agents = args.n_agents
        # k = args.k

        # split up the state tuple
        state_value, state_network = env_state

        assert state_value.shape == (n_agents, n_states)
        assert state_network.shape == (n_agents, n_agents)
        assert np.sum(np.diag(state_network)) == 0  # assume no self loops

        # reshape values and network to correct shape
        state_value = state_value.transpose(1, 0)
        messages = messages.transpose(1, 0)

        state_value = state_value.reshape((1, 1, n_states, n_agents))
        state_network = state_network.reshape((1, 1, n_agents, n_agents))
        state_message = messages.reshape((1, 1, msg_len, n_agents))

        # move matrices to GPU
        self.value = torch.Tensor(state_value).to(device)
        self.network = torch.Tensor(state_network).to(device)
        self.message = torch.Tensor(state_message).to(device)


