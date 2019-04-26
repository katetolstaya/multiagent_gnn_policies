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

class GNN(nn.Module):

    def __init__(self, layers, K):
        super(GNN, self).__init__()

        self.layers = layers
        self.n_layers = len(layers) - 1
        self.conv_layers = []

        for i in range(self.n_layers):
            m = nn.Conv2d(in_channels=K, out_channels=layers[i + 1], kernel_size=(layers[i], 1), stride=(layers[i], 1))
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
        m = nn.Conv2d(in_channels=K, out_channels=layers[1], kernel_size=(layers[0], 1), stride=(layers[0], 1))
        self.conv_layers.append(m)

        for i in range(1, self.n_layers):
            m = nn.Conv2d(in_channels=1, out_channels=layers[i + 1], kernel_size=(layers[i], 1), stride=(layers[i], 1))
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

