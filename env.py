import numpy as np
# Ok, let's follow the same ideas from the last time. Let's start by being able to run episodes in an environment.

from gym.spaces import Discrete

class DiscreteEnvironment:
    """
    Discrete states, discrete actions. We assume rewards on actions. Formally, this means:
    R_t = r(S_t, A_t). "The reward is a function of the state and the action we're in". Alternative formalizations
    include rewards on states alone, and rewards depending on the next state as well.
    """
    class TransitionMatrix:
        def __init__(self, environment, dynamics):
            self.env = environment
            self.dynamics = dynamics
            if callable(dynamics):
                # We assume that a function takes (state, action)
                self.transition_by_function = True
            else:
                self.transition_by_function = False
                if self.dynamics.shape != (len(self.env.states), len(self.env.actions), len(self.env.states)):
                    raise Exception("Transition matrix should be n_states per n_actions per n_states")

        def transition(self, state, action):
            if self.transition_by_function:
                return self.dynamics(state, action)
            else:
                return np.random.choice(self.env.states, p=self.dynamics[state, action])[0]

    def __init__(self, n_states, terminal_states, n_actions, transition_matrix, rewards):
        self.state_space = Discrete(n_states)
        self.terminal_states = np.array(terminal_states)
        self.nonterminal_states = [x for x in range(self.state_space.n) if x not in self.terminal_states]
        self.action_space = Discrete(n_actions)
        self.transition_matrix = DiscreteEnvironment.TransitionMatrix(self, transition_matrix)
        self.rewards = np.array(rewards)
        self.cur_state = np.random.choice(self.nonterminal_states)

    def step(self, action):
        next_state = self.transition_matrix.transition(self.cur_state, action)
        reward = self.rewards[self.cur_state, action]
        self.cur_state = next_state
        self.done = self.cur_state in self.terminal_states
        return self.cur_state, reward, self.done

    def reset(self):
        self.cur_state = np.random.choice(self.nonterminal_states)
        self.done = False
        return self.cur_state

    # Ok, let's implement a Builder pattern

    class Builder:
        def __init__(self):
            self._n_states = None
            self._n_actions = None

            self._terminal_states = None

            self._transition_dynamics = None
            self._rewards = None

        def set_n_states(self, n_states):
            self._n_states = n_states
            return self

        def set_terminal_states(self, list_of_terminals):
            self._terminal_states = list(list_of_terminals)
            return self

        def set_n_actions(self, n_actions):
            self._n_actions = n_actions
            return self

        def set_transition_dynamics(self, transitions):
            self._transition_dynamics = transitions
            return self

        def set_rewards(self, rewards):
            self._rewards = rewards
            return self

        def build(self):
            return DiscreteEnvironment(
                self._n_states,
                self._terminal_states,
                self._n_actions,
                self._transition_dynamics,
                self._rewards
            )
