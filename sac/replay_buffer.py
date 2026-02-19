import numpy as np

class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = int(max_size)
        self.ptr = 0
        self.current_size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size

        if self.current_size < self.max_size:
            self.current_size += 1

    def sample(self, batch_size=256):

        indices = np.random.randint(0, self.current_size, batch_size)

        batch_states = self.state[indices]
        batch_actions = self.action[indices]
        batch_rewards = self.reward[indices]
        batch_next_states = self.next_state[indices]
        batch_done = self.done[indices]

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_done