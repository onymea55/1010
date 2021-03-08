import numpy as np
import copy


class ReplayMemorySARSd():
    def __init__(self, capacity=10000, sample_size=32):
        self.capacity = capacity
        self.sample_size = sample_size
        self.memory = []

    def memorize(self, state, action, reward, state_1, term):
        index = np.random.randint(len(self.memory)) if self.memory else 0
        self.memory.insert(index, (copy.deepcopy(state), copy.deepcopy(action), copy.deepcopy(reward), copy.deepcopy(state_1), copy.deepcopy(term)))
        if len(self.memory) > self.capacity:
            self.memory = self.memory[:self.capacity]

    def sample(self):
        res = []
        for i in range(min(self.sample_size, len(self.memory))):
            res.append(copy.deepcopy(self.memory[np.random.choice(len(self.memory))]))

        return res

class ReplayMemoryTransition():
    def __init__(self, capacity=10000, sample_size=32):
        self.capacity = capacity
        self.sample_size = sample_size
        self.memory = []

    def memorize(self, action, reward, transition_grid, state_1, term):
        index = np.random.randint(len(self.memory)) if self.memory else 0
        self.memory.insert(index, (copy.deepcopy(action), copy.deepcopy(reward), copy.deepcopy(transition_grid), copy.deepcopy(state_1), copy.deepcopy(term)))
        if len(self.memory) > self.capacity:
            self.memory = self.memory[:self.capacity]

    def sample(self):
        res = []
        for i in range(min(self.sample_size, len(self.memory))):
            res.append(copy.deepcopy(self.memory[np.random.choice(len(self.memory))]))

        return res