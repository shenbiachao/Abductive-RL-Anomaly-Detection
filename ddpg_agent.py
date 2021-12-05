import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import config
import common


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dims):
        super(MLPNetwork, self).__init__()
        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        hidden_dims = [input_dim] + hidden_dims
        self.networks = []
        act_cls = nn.ReLU
        out_act_cls = nn.Identity
        for i in range(len(hidden_dims) - 1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i + 1]
            curr_network = nn.Linear(curr_shape, next_shape)
            self.networks.extend([curr_network, act_cls()])
        final_network = nn.Linear(hidden_dims[-1], out_dim)
        self.networks.extend([final_network, out_act_cls()])
        self.networks = nn.Sequential(*self.networks)

    def forward(self, input):
        return self.networks(input)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, deterministic=False, re_parameterize=True):
        super(PolicyNetwork, self).__init__()
        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        hidden_dims = [input_dim] + hidden_dims
        action_dim = config.sample_size
        self.dist_cls = torch.distributions.Normal
        self.networks = []
        act_cls = nn.ReLU
        out_act_cls = nn.Identity
        for i in range(len(hidden_dims)-1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i+1]
            curr_network = nn.Linear(curr_shape, next_shape)
            self.networks.extend([curr_network, act_cls()])
        if re_parameterize:
            # output mean and std for re-parametrization
            final_network = nn.Linear(hidden_dims[-1], action_dim * 2)
        else:
            # output mean and let std to be optimizable
            final_network = nn.Linear(hidden_dims[-1], action_dim)
            log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.networks.extend([final_network, out_act_cls()])
        self.networks = nn.Sequential(*self.networks)
        # action rescaler
        self.action_scale = torch.FloatTensor([(config.upper_bound - config.lower_bound) / 2.] * action_dim)
        self.action_bias = torch.FloatTensor([(config.upper_bound + config.lower_bound) / 2.] * action_dim)

        self.action_dim = action_dim
        self.noise = torch.Tensor(action_dim)
        self.deterministic = deterministic
        self.re_parameterize = re_parameterize

    def forward(self, state):
        out = self.networks(state)
        if self.re_parameterize:
            # action_mean, action_log_std = torch.split(out, [self.action_dim, self.action_dim], dim=1)
            action_mean = out[:, :self.action_dim]
            action_log_std = out[:, self.action_dim:]
            if self.deterministic:
                return action_mean, None
            else:
                return action_mean, action_log_std
        else:
            return out, None

    def sample(self, state):
        action_mean_raw, action_log_std_raw = self.forward(state)
        if self.deterministic:
            action_mean_scaled = torch.tanh(action_mean_raw) * self.action_scale + self.action_bias
            # noise = self.noise.normal_(0., std=0.1)
            # noise = noise.clamp(-0.25, 0.25)
            # action = action_mean + noise
            action = action_mean_scaled
            return action, torch.tensor(0.), action_mean_scaled
        else:
            if self.re_parameterize:
                action_std_raw = action_log_std_raw.exp()
                dist = self.dist_cls(action_mean_raw, action_std_raw)
                mean_sample_raw = dist.rsample()
                action = torch.tanh(mean_sample_raw) * self.action_scale + self.action_bias
                log_prob_raw = dist.log_prob(mean_sample_raw)
                log_prob_stable = log_prob_raw - torch.log(
                    self.action_scale * (1 - torch.tanh(mean_sample_raw).pow(2)) + 1e-6)
                log_prob = log_prob_stable.sum(1, keepdim=True)
                action_mean_scaled = torch.tanh(action_mean_raw) * self.action_scale + self.action_bias
                return action, log_prob, action_mean_scaled, {
                    "action_std": action_std_raw,
                    "pre_tanh_value": mean_sample_raw  # todo: check scale
                }

            else:
                dist = self.dist_cls(action_mean_raw, torch.exp(self.log_std))
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
                return action, log_prob, action_mean_raw

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(PolicyNetwork, self).to(device)


class DDPGAgent(torch.nn.Module):
    def __init__(self, obs_dim):
        super(DDPGAgent, self).__init__()
        obs_dim = obs_dim
        action_dim = config.sample_size

        # initilze networks
        self.q_network = MLPNetwork(obs_dim + action_dim, 1, config.agent_hidden_dims)
        self.target_q_network = MLPNetwork(obs_dim + action_dim, 1, config.agent_hidden_dims)
        self.policy_network = PolicyNetwork(obs_dim, config.agent_hidden_dims)
        self.target_policy_network = PolicyNetwork(obs_dim, config.agent_hidden_dims)

        # sync network parameters
        common.hard_update_network(self.q_network, self.target_q_network)
        common.hard_update_network(self.policy_network, self.target_policy_network)

        # pass to util.device
        self.q_network = self.q_network.to(config.device)
        self.target_q_network = self.target_q_network.to(config.device)
        self.policy_network = self.policy_network.to(config.device)
        self.target_policy_network = self.target_policy_network.to(config.device)

        # register networks
        self.networks = {
            'q_network': self.q_network,
            'target_q_network': self.target_q_network,
            'policy_network': self.policy_network,
            'target_policy_network': self.target_policy_network
        }

        # initialize optimizer
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.rl_learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=config.rl_learning_rate)

        # hyper-parameters
        self.gamma = config.gamma
        self.tot_update_count = 0
        self.update_target_network_interval = config.update_target_network_interval
        self.target_smoothing_tau = config.target_smoothing_tau

    def update(self, data_batch):
        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = data_batch

        curr_state_q_value = self.q_network(torch.cat([obs_batch, action_batch], dim=1))

        new_curr_state_action, new_curr_state_log_pi, _, _ = self.policy_network.sample(obs_batch)
        next_state_action, next_state_log_pi, _, _ = self.target_policy_network.sample(next_obs_batch)

        new_curr_state_q_value = self.q_network(torch.cat([obs_batch, new_curr_state_action], dim=1))

        next_state_q_value = self.target_q_network(torch.cat([next_obs_batch, next_state_action], dim=1))
        target_q = reward_batch + self.gamma * (1. - done_batch) * next_state_q_value

        # compute q loss
        q_loss = F.mse_loss(curr_state_q_value, target_q.detach())

        q_loss_value = q_loss.detach().cpu().numpy()
        self.q_optimizer.zero_grad()
        q_loss.backward()

        # compute policy loss
        policy_loss = new_curr_state_q_value.mean()
        policy_loss_value = policy_loss.detach().cpu().numpy()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        self.q_optimizer.step()
        self.policy_optimizer.step()

        self.tot_update_count += 1

        # update target network
        self.try_update_target_network()

        return {
            "loss/q": q_loss_value,
            "loss/policy": policy_loss_value,
        }

    def try_update_target_network(self):
        if self.tot_update_count % self.update_target_network_interval == 0:
            common.soft_update_network(self.q_network, self.target_q_network, self.target_smoothing_tau)
            common.soft_update_network(self.policy_network, self.target_policy_network, self.target_smoothing_tau)

    def select_action(self, state, deterministic=False):
        action, log_prob, mean, std = self.policy_network.sample(state)
        if deterministic:
            return mean.detach().cpu().numpy()[0], log_prob
        else:
            return action.detach().cpu().numpy()[0], log_prob
