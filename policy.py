import numpy as np
import gym
from gym.spaces import Discrete, Box # Let's deal with those for now.

class DiscretePolicy:
    """
    This is the first implementation of a Policy abstraction that I've done in my life.
    
    """
    def __init__(self, state_space, action_space):
        if isinstance(state_space, Discrete):
            self.v_est = np.zeros(state_space.n)
            if isinstance(action_space, Discrete):
                self.q_est = np.zeros((state_space.n, action_space.n))
        elif isinstance(state_space, Box):
            raise NotImplemented('Support for continuous state spaces not implemented')
        self.policy = np.ones((state_space.n, action_space.n)) * (1 / action_space.n)
        self.action_space = action_space
        self.state_space = state_space

    def step(self, state):
        return np.random.choice(np.arange(self.action_space.n, dtype=np.int), 1, p=self.policy[state])[0]

