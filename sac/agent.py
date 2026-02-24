import torch 
import torch.nn as nn
from .actor import Actor
from .critic import Critic
from .replay_buffer import ReplayBuffer


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2


    def update(self, replay_buffer, batch_size):
        states, action, rewards, next_states, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log = self.actor(states)

            target_q1 = self.target_critic1(states, next_action)
            target_q2 = self.target_critic1(states, next_action)
            
            min_q = torch.min(target_q1, target_q2)
            y = rewards + self.gamma(min_q - self.alpha * next_log)
