import numpy as np
import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from .actor import Actor
from .critic import Critic
from .replay_buffer import ReplayBuffer
from .utils import scale_action_to_env


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, writer=None):
        self.writer = writer
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.critic_loss = nn.MSELoss()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005

        # Auto entropy tuning
        self.target_entropy = -0.5 * float(action_dim)  # -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.episode = 0
    
    def select_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.actor(state_tensor, deterministic=deterministic, with_logprob=False)
        return action.detach().cpu().squeeze(0).numpy()  # returns [-1, 1]

    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, done = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states) 
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        done = torch.FloatTensor(done)

        with torch.no_grad():
            next_action, next_log = self.actor(next_states)

            target_q1 = self.target_critic1(next_states, next_action)
            target_q2 = self.target_critic2(next_states, next_action)
            
            min_q = torch.min(target_q1, target_q2)
            y = rewards + self.gamma * (1 - done) * (min_q - self.alpha * next_log)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        loss_critic1 = self.critic_loss(q1, y)
        loss_critic2 = self.critic_loss(q2, y)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        
        loss_critic1.backward()
        loss_critic2.backward()

        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        action, log_prob = self.actor(states)

        q1 = self.critic1(states, action)
        q2 = self.critic2(states, action)

        min_q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_prob - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        if self.writer is not None:
            self.writer.add_scalar('Loss/actor', actor_loss.item(), self.episode)
            self.writer.add_scalar('Loss/critic1', loss_critic1.item(), self.episode)
            self.writer.add_scalar('Loss/critic2', loss_critic2.item(), self.episode)
            self.writer.add_scalar('Alpha', self.alpha, self.episode)

        with torch.no_grad():
            for target_param, param in zip(self.target_critic1.parameters(),
                                        self.critic1.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for target_param, param in zip(self.target_critic2.parameters(),
                                        self.critic2.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            
        self.episode += 1
        

