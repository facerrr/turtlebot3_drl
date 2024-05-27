import torch
import random
from collections import deque
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from common.ounoise import OUNoise
from common.replay_buffer import ReplayBuffer
    
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = torch.nn.Linear(state_dim, hidden_dim)
        self.layer_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = torch.nn.Linear(hidden_dim, action_dim)
        nn.init.kaiming_normal_(self.layer_1.weight, mode='fan_in', nonlinearity='relu')
        self.layer_1.bias.data.fill_(0.01)
        nn.init.kaiming_normal_(self.layer_2.weight, mode='fan_in', nonlinearity='relu')
        self.layer_2.bias.data.fill_(0.01)
        nn.init.xavier_normal_(self.layer_3.weight)
        self.layer_3.bias.data.fill_(0.01)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x
    
class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.layer_1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = torch.nn.Linear(hidden_dim, 1)
        nn.init.kaiming_normal_(self.layer_1.weight, mode='fan_in', nonlinearity='relu')
        self.layer_1.bias.data.fill_(0.01)
        nn.init.kaiming_normal_(self.layer_2.weight, mode='fan_in', nonlinearity='relu')
        self.layer_2.bias.data.fill_(0.01)
        nn.init.xavier_normal_(self.layer_3.weight)
        self.layer_3.bias.data.fill_(0.01)

    def forward(self, x, u):
        x = F.relu(self.layer_1(torch.cat([x, u], 1)))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x
    
class DDPG:
    def __init__(self, state_dim, action_dim, max_action, device):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sigma = 0.1
        self.gamma = 0.99
        self.tau = 0.003
        self.learning_rate = 0.003
        self.device = device
        self.hidden_dim = 512
        self.noise = OUNoise(action_space=action_dim, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)
  
        self.batch_size = 256
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim, max_action).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.hidden_dim, max_action).to(self.device)

        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def __name__(self):
        return "DDPG"

    def select_action(self, state):
        # list[n_states]-->tensor[1,n_states]-->gpu
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)[0].cpu().data.numpy().flatten()
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
    
        state = torch.FloatTensor(batch_states).to(self.device)
        action = torch.FloatTensor(batch_actions).to(self.device)
        reward = torch.FloatTensor(batch_rewards).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(batch_next_states).to(self.device)
        done = torch.FloatTensor(batch_dones).to(self.device).unsqueeze(1)

        with torch.no_grad():
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (1 - done) * self.gamma * target_Q

        curretn_Q = self.critic(state, action)
        critic_loss = F.mse_loss(curretn_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        # Freeze critic networks
        for param in self.critic.parameters():
            param.requires_grad = False
        
        #compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for param in self.critic.parameters():
            param.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
