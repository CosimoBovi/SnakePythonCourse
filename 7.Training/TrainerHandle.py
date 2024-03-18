import random
import numpy as np
from collections import deque
from SnakeModel import ActionResult

MAX_MEMORY = 1000000
BATCH_SIZE = 1000

class TrainerHandle:

    def __init__(self, trainer):
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.trainer=trainer
        self.model = self.trainer.model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    # per velocità di esecuzione short memeory è solo uno
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)