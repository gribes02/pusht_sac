import torch
import torch.nn as nn  


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim)
        )
    
    def forward(self, state, deterministic=False, with_logprob=True):
        x = self.mlp(state)
        mu, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        if deterministic:
            z = mu
        else:
            z = normal.rsample()
        action = torch.tanh(z)
        if not deterministic and with_logprob:
            log_prob = normal.log_prob(z)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        else: 
            log_prob = None
        
        return action, log_prob
    