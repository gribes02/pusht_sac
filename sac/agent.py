import torch 
import torch.nn as nn
from .actor import Actor
from .critic import Critic
from .replay_buffer import ReplayBuffer


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
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
        self.alpha = 0.2


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
            
        

