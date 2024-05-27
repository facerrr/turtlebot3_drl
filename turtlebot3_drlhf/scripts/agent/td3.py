from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random
from common.replay_buffer import ReplayBuffer

CAPACITY = 1000000

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    
class TD3:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = 256
        self.batch_size = 256
        self.GAMMA = 0.99
        self.TAU = 0.005
        self.lr = 3e-4
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.device = device
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, self.hidden_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.replayBuffer = ReplayBuffer()


    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        action = action.tolist()
        for i in range(self.action_dim):
            action[i] = np.clip(action[i], -1, 1)
        return action
    
    def update(self, iterations):
        if len(self.replayBuffer) < self.batch_size:
            return
        av_Q = 0
        max_Q = -np.inf
        av_loss = 0
        for it in range(iterations):
            (batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones) = self.replayBuffer.sample(self.batch_size)
            state = torch.FloatTensor(batch_states).to(self.device)
            action = torch.FloatTensor(batch_actions).to(self.device)
            reward = torch.FloatTensor(batch_rewards).unsqueeze(1).to(self.device)
            next_state = torch.FloatTensor(batch_next_states).to(self.device)
            done = torch.FloatTensor(batch_dones).unsqueeze(1).to(self.device)

            next_action = self.actor_target(next_state)

            noise = torch.Tensor(batch_actions).data.normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            target_Q = reward + ((1 - done) * self.GAMMA * target_Q).detach()

            current_Q1, current_Q2 = self.critic(state, action)
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it%self.policy_freq == 0:
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_loss = -torch.mean(actor_grad)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
                
            av_loss += loss.item()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")
    
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))

        
