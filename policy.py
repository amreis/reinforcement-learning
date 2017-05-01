import numpy as np
from gym.spaces import Discrete, Box # Let's deal with those for now.

class Policy:
    def __init__(self, state_space, action_space):
        pass

    def step(self, state):
        pass


class DiscretePolicy(Policy):
    """
    This is the first implementation of a Policy abstraction that I've done in my life.

    """
    def __init__(self, state_space, action_space):
        super(Policy, self).__init__()
        self.policy = np.ones((state_space.n, action_space.n)) * (1 / action_space.n)
        self.action_space = action_space
        self.state_space = state_space

    def step(self, state):
        return np.random.choice(np.arange(self.action_space.n, dtype=np.int), p=self.policy[state])


class ContinuousStatePolicy(Policy):
    def __init__(self, state_space, action_space):
        super(ContinuousStatePolicy, self).__init__()
        self.action_space = action_space
        self.state_space = state_space

    def step(self, state):
        return self.action_space.sample()