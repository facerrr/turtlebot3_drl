import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)

        nn.init.kaiming_normal_(self.l1.weight, mode='fan_in', nonlinearity='relu')
        self.l1.bias.data.fill_(0.01)
        nn.init.kaiming_normal_(self.l2.weight, mode='fan_in', nonlinearity='relu')
        self.l2.bias.data.fill_(0.01)
        nn.init.xavier_normal_(self.mean_layer.weight)
        self.mean_layer.bias.data.fill_(0.01)
        nn.init.xavier_normal_(self.log_std_layer.weight)
        self.log_std_layer.bias.data.fill_(0.01)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        dist = Normal(mean, std)
        if deterministic: 
            a = mean
        else:
            a = dist.rsample() 

        if with_logprob: 
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * torch.tanh(a) 

        return a, log_pi
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)

        nn.init.kaiming_normal_(self.l1.weight, mode='fan_in', nonlinearity='relu')
        self.l1.bias.data.fill_(0.01)
        nn.init.kaiming_normal_(self.l2.weight, mode='fan_in', nonlinearity='relu')
        self.l2.bias.data.fill_(0.01)
        nn.init.kaiming_normal_(self.l3.weight, mode='fan_in', nonlinearity='relu')
        self.l3.bias.data.fill_(0.01)

        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)

        nn.init.kaiming_normal_(self.l4.weight, mode='fan_in', nonlinearity='relu')
        self.l4.bias.data.fill_(0.01)
        nn.init.kaiming_normal_(self.l5.weight, mode='fan_in', nonlinearity='relu')
        self.l5.bias.data.fill_(0.01)
        nn.init.kaiming_normal_(self.l6.weight, mode='fan_in', nonlinearity='relu')
        self.l6.bias.data.fill_(0.01)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2
    
class SAC:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_width = 512  
        self.batch_size = 256
        self.GAMMA = 0.99  
        self.TAU = 0.0001 
        self.lr = 3e-4
        self.adaptive_alpha = True
        self.device = device   

        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2

        self.actor = Actor(state_dim, action_dim, self.hidden_width, self.max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def __name__(self):
        return "SAC"

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state, deterministic, False)[0].cpu().data.numpy().flatten()
        action = np.clip(action.tolist(), -1.0, 1.0)
        return action
    
    def update(self, replay_buffer, human_data):
        if len(replay_buffer) < self.batch_size:
            return

        if human_data is None:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(self.batch_size)
        else:
            size1 = int(self.batch_size * 0.8)
            size2 = self.batch_size - size1

            bs_1, ba_1, br_1, bns_1, bd_1 = replay_buffer.sample(size1)
            bs_2, ba_2, br_2, bns_2, bd_2 = human_data.sample(size2)

            batch_states = np.concatenate((bs_1, bs_2), axis=0)
            batch_actions = np.concatenate((ba_1, ba_2), axis=0)
            batch_rewards = np.concatenate((br_1, br_2), axis=0)
            batch_next_states = np.concatenate((bns_1, bns_2), axis=0)
            batch_dones = np.concatenate((bd_1, bd_2), axis=0)

        batch_s = torch.FloatTensor(batch_states).to(self.device)
        batch_a = torch.FloatTensor(batch_actions).to(self.device)
        batch_r = torch.FloatTensor(batch_rewards).to(self.device).unsqueeze(1)
        batch_ns = torch.FloatTensor(batch_next_states).to(self.device)
        batch_d = torch.FloatTensor(batch_dones).to(self.device).unsqueeze(1)

        with torch.no_grad():
            batch_a_, log_pi = self.actor(batch_ns)
            q1_next, q2_next = self.critic_target(batch_ns, batch_a_)
            alpha = self.alpha.to(self.device)
            q_next = torch.min(q1_next, q2_next) - alpha * log_pi
            target_q = batch_r + self.GAMMA * (1 - batch_d) * q_next
        
        q1, q2 = self.critic(batch_s, batch_a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        for params in self.critic.parameters():
            params.requires_grad = False

        a, log_pi = self.actor(batch_s)
        q1, q2 = self.critic(batch_s, a)

        actor_loss = (alpha * log_pi - torch.min(q1, q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        for params in self.critic.parameters():
            params.requires_grad = True
        
        if self.adaptive_alpha:
            log_alpha = self.log_alpha.to(self.device)
            alpha_loss = -(log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")
    
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))

        