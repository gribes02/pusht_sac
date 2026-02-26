import os
import gymnasium as gym
import gym_pusht
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from sac.agent import SACAgent
from sac.replay_buffer import ReplayBuffer
from sac.utils import normalize_state, scale_action_to_env, scale_action_to_tanh


class Train():
    def __init__(self, state_dim, action_dim, env,
                 num_episodes=3000, checkpoint_dir='checkpoints', resume=True):

        self.num_episodes = num_episodes
        self.total_steps = 0
        self.start_steps = 5000
        self.max_steps = 300
        self.batch_size = 256
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = 100       # save every N episodes
        self.env = env
        self.writer = SummaryWriter(log_dir='logs')
        self.agent = SACAgent(state_dim, action_dim, writer=self.writer)
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        self.reward_window = deque(maxlen=100)  # rolling mean over last 100 eps
        self.start_episode = 0

        if resume and os.path.exists(f'{checkpoint_dir}/checkpoint.pt'):
            self.agent.load(checkpoint_dir)
            self.total_steps = self.start_steps + self.agent.episode
            print(f"Resuming — agent update step: {self.agent.episode}")


    def train(self):
        for episode in range(self.start_episode, self.num_episodes):

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
                self.total_steps += 1

                if self.replay_buffer.current_size > self.batch_size:
                    self.agent.update(self.replay_buffer, self.batch_size)

                if done:
                    break

            self.reward_window.append(episode_reward)
            mean_reward = np.mean(self.reward_window)

            print(f"Episode {episode:4d} | Reward: {episode_reward:8.2f} | Mean-100: {mean_reward:8.2f} | Steps: {self.total_steps}")
            self.writer.add_scalar('Reward/episode', episode_reward, episode)
            self.writer.add_scalar('Reward/mean_100', mean_reward, episode)

            if (episode + 1) % self.checkpoint_every == 0:
                self.agent.save(self.checkpoint_dir)
                print(f"  >> Checkpoint saved at episode {episode + 1}")

        # Final save
        self.agent.save(self.checkpoint_dir)
        print("Training complete — final checkpoint saved.")
        self.writer.close()
        self.env.close()


if __name__ == "__main__":
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    state_dim = 6
    action_dim = 2

    trainer = Train(
        state_dim, action_dim, env,
        num_episodes=3000,      # increase for better policy
        checkpoint_dir='checkpoints',
        resume=True,            # set False to train from scratch
    )
    trainer.train()