import gymnasium as gym
import gym_pusht
from torch.utils.tensorboard import SummaryWriter
from sac.agent import SACAgent 
from sac.replay_buffer import ReplayBuffer
from sac.utils import normalize_state, scale_action_to_env, scale_action_to_tanh

class Train():
    def __init__(self, state_dim, action_dim, env):
        
        self.num_episodes = 500
        self.total_steps = 0
        self.start_steps = 5000
        self.max_steps = 300
        self.batch_size = 256
        self.env =  env
        self.writer = SummaryWriter(log_dir='logs')
        self.agent = SACAgent(state_dim, action_dim, writer=self.writer)
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)


    def train(self):
        for episode in range(self.num_episodes):

            episode_reward = 0 

            state, _ = self.env.reset()
            state = normalize_state(state)

            for step in range(self.max_steps):
                if self.total_steps < self.start_steps:
                    env_action = self.env.action_space.sample()  # [0, 512]
                    action = scale_action_to_tanh(env_action)    # [-1, 1] for buffer
                else:
                    action = self.agent.select_action(state)      # [-1, 1]
                    env_action = scale_action_to_env(action)      # [0, 512] for env

                next_state, reward, terminated, truncated, _ = self.env.step(env_action)
                next_state = normalize_state(next_state)

                episode_reward += reward

                # Scale reward to stabilize critic training
                scaled_reward = reward / 100.0

                done = terminated or truncated
                self.replay_buffer.add(state, action, scaled_reward, next_state, done)

                state = next_state
                self.total_steps +=1

                if self.replay_buffer.current_size > self.batch_size:
                    self.agent.update(self.replay_buffer, self.batch_size)
                
                if done:
                    break

            print(f"Episode {episode}, Reward: {episode_reward}")
            self.writer.add_scalar('Reward/episode', episode_reward, episode)

        self.writer.close()
        self.env.close()

if __name__ == "__main__":
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    state_dim = 6
    action_dim = 2

    trainer = Train(state_dim, action_dim, env)
    trainer.train()