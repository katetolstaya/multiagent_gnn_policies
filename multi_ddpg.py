import sys
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class TiedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TiedLinear, self).__init__()
        self.n_agents = 30
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        repeated_weight = self.weight.repeat(self.n_agents, self.n_agents) #.flatten()
        repeated_bias = self.bias.repeat(1, self.n_agents) #.flatten()
        return F.linear(input, repeated_weight, repeated_bias)


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()

        self.action_space = action_space
        self.n_agents = 30
        self.num_outputs = 2 #int(action_space.shape[0]/self.n_agents)
        self.num_inputs = 12 #int(num_inputs/self.n_agents)

        self.linear1 = TiedLinear(self.num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size*self.n_agents)

        self.linear2 = TiedLinear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size*self.n_agents)

        self.mu = TiedLinear(hidden_size, self.num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        mu = F.tanh(self.mu(x))
        return mu
        # mu = []
        # for i in range(self.n_agents):
        #     x = inputs[:,i*self.num_inputs:(i+1)*self.num_inputs]
        #     x = self.linear1(x)
        #     x = self.ln1(x)
        #     x = F.relu(x)
        #     x = self.linear2(x)
        #     x = self.ln2(x)
        #     x = F.relu(x)
        #     mu.append(torch.tanh(self.mu(x)))
        # return torch.cat(mu,1)

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()

        self.action_space = action_space
        self.n_agents = 30
        self.num_outputs = 2 #int(action_space.shape[0]/self.n_agents)
        self.num_inputs = 12 #int(num_inputs/self.n_agents)

        self.linear1 = TiedLinear(self.num_inputs + self.num_outputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size*self.n_agents)

        self.linear2 = TiedLinear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size*self.n_agents)

        self.V = TiedLinear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x = torch.cat((inputs, actions), 1)
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        V = self.V(x)
        return V

        # V = torch.zeros([1], dtype=torch.float32)
        # V = []
        # for i in range(self.n_agents):
        #     state = inputs[:,(i*self.num_inputs):((i+1)*self.num_inputs)]
        #     action = actions[:,(i*self.num_outputs):((i+1)*self.num_outputs)]
        #     x = torch.cat((state,action),1)
        #     x = self.linear1(x)
        #     x = self.ln1(x)
        #     x = F.relu(x)
        #     x = self.linear2(x)
        #     x = self.ln2(x)
        #     x = F.relu(x)
        #     V.append(self.V(x)) 
        # return torch.sum(torch.cat(V,1),1) # TODO can use other operations like max or min

class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, device):
        self.device = device
        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)


        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None, param_noise=None):
        self.actor.eval()
        if param_noise is not None: 
            mu = self.actor_perturbed((Variable(state)))
        else:
            mu = self.actor((Variable(state)))

        self.actor.train()
        mu = mu.data

        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise()).to(self.device)

        return mu.clamp(-1, 1)


    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state)).to(self.device)
        action_batch = Variable(torch.cat(batch.action)).to(self.device)
        reward_batch = Variable(torch.cat(batch.reward)).to(self.device)
        mask_batch = Variable(torch.cat(batch.mask)).to(self.device)
        next_state_batch = Variable(torch.cat(batch.next_state)).to(self.device)
        
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic((state_batch),self.actor((state_batch)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            param += torch.randn(param.shape) * param_noise.current_stddev

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix) 
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location='cpu'))
            self.actor = self.actor.to(self.device)
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path, map_location='cpu'))
            self.critic = self.critic.to(self.device)