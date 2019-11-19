import torch
import torch.nn as nn
from torch.distributions import Categorical


class Message(nn.Module):

    def __init__(self, n_s, msg_len, hidden_layers):
        """
        The policy network is allowed to have only one aggregation operation due to communication latency, but we can
        have any number of hidden layers to be executed by each agent individually.
        :param n_s: number of MDP states per agent
        :param n_a: number of MDP actions per agent
        :param hidden_layers: list of ints that will determine the width of each hidden layer
        :param k: aggregation filter length
        :param ind_agg: before which MLP layer index to aggregate
        """
        super(Message, self).__init__()
        self.msg_len = msg_len
        self.n_s = n_s
        self.msg_len = msg_len

        if msg_len > 0:
            self.layers = [n_s + msg_len + msg_len] + hidden_layers + [msg_len]
            # self.layers = [n_s] + hidden_layers + [n_a + msg_len]
            # self.layers = [n_s + msg_len] + hidden_layers + [n_a + msg_len]
            self.n_layers = len(self.layers) - 1

            self.conv_layers = []

            for i in range(0, self.n_layers):
                m = nn.Conv2d(in_channels=self.layers[i], out_channels=self.layers[i + 1], kernel_size=(1, 1),
                              stride=(1, 1))

                self.conv_layers.append(m)

            self.conv_layers = torch.nn.ModuleList(self.conv_layers)

    def forward(self, value, network, message):
        """
        The policy relies on delayed information from neighbors. During training, the full history for k time steps is
        necessary.
        :param value:
        :param network:
        :param message:
        :return:
        """
        batch_size = value.shape[0]
        unroll_len = value.shape[1]
        n_agents = value.shape[3]
        assert network.shape[0] == batch_size
        assert network.shape[2] == n_agents
        assert network.shape[3] == n_agents

        assert value.shape[2] == self.n_s

        if self.msg_len > 0:
            current_message = message[:, 0, :, :]
            current_message = current_message.view((batch_size, 1, self.msg_len, n_agents))

            for l in range(unroll_len):
                current_network = network[:, l, :, :].view((batch_size, 1, n_agents, n_agents))
                passed_messages = torch.matmul(current_message, current_network)  # B, F, N
                current_values = value[:, l, :, :].view((batch_size, 1, self.n_s, n_agents))  # B, F, N
                x = torch.cat((current_values, passed_messages, current_message), 2)  # cat in features dim

                if l < unroll_len - 1:
                    x = x.permute(0, 2, 1, 3)  # now (B,F,K,N)

                    for i in range(self.n_layers):
                        # print(x.size())
                        x = self.conv_layers[i](x)
                        x = torch.tanh(x)

                    x = x.permute(0, 2, 1, 3)  # now (B,K,F,N)
                    current_message = x[:, :, -self.msg_len:, :].view((batch_size, 1, self.msg_len, n_agents))
                else:
                    current_message = x.view((batch_size, 1, self.n_s + self.msg_len + self.msg_len, n_agents))
        else:
            current_message = value[:, -1, :, :].view((batch_size, 1, self.n_s, n_agents))

        return current_message


# TODO: how to deal with bounded/unbounded action spaces?? Should I always assume bounded actions?


class Actor(nn.Module):

    def __init__(self, n_s, n_a, msg_len, hidden_layers):
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
        self.msg_len = msg_len
        self.n_s = n_s
        self.n_a = n_a
        self.msg_len = msg_len
        self.layers = [n_s + msg_len + msg_len] + hidden_layers + [n_a]
        self.n_layers = len(self.layers) - 1

        self.conv_layers = []

        for i in range(0, self.n_layers):
            m = nn.Conv2d(in_channels=self.layers[i], out_channels=self.layers[i + 1], kernel_size=(1, 1),
                          stride=(1, 1))

            self.conv_layers.append(m)

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        self.layer_norms = []
        for i in range(self.n_layers - 1):
            m = nn.GroupNorm(self.layers[i + 1], self.layers[i + 1])
            self.layer_norms.append(m)

        self.layer_norms = torch.nn.ModuleList(self.layer_norms)



    def forward(self, value_msg, eval_actions=None):
        """
        The policy relies on delayed information from neighbors. During training, the full history for k time steps is
        necessary.
        :param value:
        :param network:
        :param message:
        :return:
        """
        batch_size = value_msg.shape[0]
        n_agents = value_msg.shape[3]
        assert value_msg.shape[2] == self.n_s + self.msg_len * 2

        x = value_msg[:, 0, :, :].view((batch_size, 1, self.n_s + self.msg_len * 2, n_agents))  # B, F, N
        x = x.permute(0, 2, 1, 3)  # now (B,F,K,N)
        for i in range(self.n_layers):
            # print(x.size())
            x = self.conv_layers[i](x)
            if i < self.n_layers - 1:
                x = self.layer_norms[i](x)
            x = torch.tanh(x)

        x = x.permute(0, 3, 1, 2)  # now (B,N,F,K)
        logits = x.contiguous().view((batch_size * n_agents, self.n_a))
        dist = Categorical(logits=logits)

        actions = dist.sample()
        actions = actions.view((batch_size, 1, 1, n_agents))

        if eval_actions is not None:
            # eval_actions = eval_actions.contiguous().view((batch_size * n_agents, 1))
            # log_probs = torch.diag(dist.log_prob(eval_actions))
            # log_probs = log_probs.view((batch_size, 1, 1, n_agents))

            eval_actions = eval_actions.view((batch_size * n_agents, 1)).long()
            all_agents = torch.arange(batch_size * n_agents).view((-1, 1))
            log_probs = dist.probs[all_agents, eval_actions].log()
            log_probs = log_probs.view((batch_size, 1, 1, n_agents))
        else:
            log_probs = None

        dist_entropy = dist.entropy().mean()

        return actions, log_probs, dist_entropy



class Critic(nn.Module):

    def __init__(self, n_s, msg_len, hidden_layers):
        """
        The actor network is centralized, so it can have any number of [GSO -> Linearity -> Activation] layers
        If there's a lot of layers here, it doesn't matter if we have a non-linearity or GSO first (test this).
        :param n_s: number of MDP states per agent
        :param n_a: number of MDP actions per agent
        :param hidden_layers: list of ints that will determine the width of each hidden layer
        :param k: aggregation filter length
        """
        super(Critic, self).__init__()

        self.msg_len = msg_len
        self.n_s = n_s
        self.msg_len = msg_len
        self.layers = [n_s + msg_len + msg_len] + hidden_layers + [1]
        self.n_layers = len(self.layers) - 1

        self.conv_layers = []

        self.use_layer_norm = False

        for i in range(0, self.n_layers):
            m = nn.Conv2d(in_channels=self.layers[i], out_channels=self.layers[i + 1], kernel_size=(1, 1),
                          stride=(1, 1))

            self.conv_layers.append(m)

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        self.layer_norms = []
        for i in range(self.n_layers - 1):
            m = nn.GroupNorm(self.layers[i + 1], self.layers[i + 1])
            self.layer_norms.append(m)

        self.layer_norms = torch.nn.ModuleList(self.layer_norms)

    def forward(self, value_msg):
        """
        The policy relies on delayed information from neighbors. During training, the full history for k time steps is
        necessary.
        :param value:
        :param network:
        :param message:
        :return:
        """
        batch_size = value_msg.shape[0]
        n_agents = value_msg.shape[3]

        assert value_msg.shape[2] == self.n_s + self.msg_len * 2

        value_msg = value_msg[:, 0, :, :].view((batch_size, 1, self.n_s + self.msg_len * 2, n_agents))
        x = value_msg
        x = x.permute(0, 2, 1, 3)  # now (B,F,K,N)

        for i in range(self.n_layers):
            x = self.conv_layers[i](x)

            if i < self.n_layers - 1:  # no tanh for the last layer
                x = self.layer_norms[i](x)
                x = torch.tanh(x)

        x = x.permute(0, 2, 1, 3)  # now (B,K,F,N), but doesn't do anything since K=F=1
        q_values = x.view((batch_size, 1, 1, n_agents))

        return q_values


