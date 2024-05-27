from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)   
        self.max_size = size

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):

        batchsize = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batchsize)
        states = np.float32([array[0] for array in batch])
        actions = np.float32([array[1] for array in batch])
        rewards = np.float32([array[2] for array in batch])
        next_states = np.float32([array[3] for array in batch])
        dones = np.float32([array[4] for array in batch])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
