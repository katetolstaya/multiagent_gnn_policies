import torch
import torch.nn as nn
import numpy as np

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

            if i > 0 or self.gso_first:  # After GSO is applied, the data has shape[1]=K channels
                in_channels = k
            else:
                in_channels = 1

            m = nn.Conv2d(in_channels=in_channels, out_channels=self.layers[i + 1], kernel_size=(self.layers[i], 1),
                          stride=(self.layers[i], 1))
            self.conv_layers.append(m)
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        self.layer_norms = []
        for i in range(self.n_layers-1):
            m = nn.GroupNorm(self.layers[i+1], self.layers[i+1])
            self.layer_norms.append(m)

        self.layer_norms = torch.nn.ModuleList(self.layer_norms)

    def forward(self, states, actions, gso):
        """
        Evaluate the critic network
        :param states: Current states of shape (B,1,n_s,N), where B = # batches, n_s = # features, N = # agents
        :param actions: Current actions of shape (B,1,n_a,N), where B = # batches, n_a = # features, N = # agents
        :param gso: Current GSO = [I, A_t, A_t^2,..., A_t^K-1] of shape (B,K,N,N)
        :return:
        """

        # check input size, which is critical for correct GSO application
        batch_size = np.shape(states)[0]
        n_agents = states.shape[3]

        assert batch_size == actions.shape[0]
        assert batch_size == gso.shape[0]
        assert states.shape[1] == 1
        assert actions.shape[2] == self.n_a
        assert actions.shape[3] == n_agents
        assert states.shape[2] == self.n_s

        assert gso.shape[1] == self.k
        assert gso.shape[2] == n_agents
        assert gso.shape[3] == n_agents

        x = torch.cat((states, actions), 2)

        # GSO -> Linearity -> Activation
        for i in range(self.n_layers):

            if i > 0 or self.gso_first:
                x = torch.matmul(x, gso)  # GSO

            x = self.conv_layers[i](x)  # Linear layer

            if i < self.n_layers - 1:  # last layer needs no relu()
                x = self.layer_norms[i](x)
                x = F.relu(x)

            x = x.view((batch_size, 1, self.layers[i + 1], n_agents))

        x = x.view((batch_size, 1, n_agents))  # now size (B, 1, N)

        return x  # size of x here is batch_size x output_layer x 1 x n_agents