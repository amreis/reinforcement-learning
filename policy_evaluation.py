"""
The idea is to implement TD and MonteCarlo methods to evaluate policies
"""
import numpy as np
from util import LinearApproximator

class MonteCarlo:
    @staticmethod
    def state_value_eval(env, policy,
                         discount=0.999,
                         learning_rate=0.01,
                         n_iter=1000,
                         print_every=None):
        """
        This is EVERY-VISIT Monte-Carlo
        :param env: An Environment that we can reset(), step() and get observations and
                    reward information.
        :param policy: A strategy for behaving in an Environment. Should have a step()
                    method that returns an action given state information.
        :param discount: Discount factor for the MDP
        :param learning_rate: The amount we will shift towards an error direction.
        :param n_iter: Number of episodes to run this algorithm for
        :param print_every: Print the current estimate of values every X iterations
        :return: The State-Value function that shows the average return we'll have starting
                 in each one of the states of this MDP
        """
        state_values = [0.0 for _ in range(env.state_space.n)]

        for episode in range(n_iter):
            done = False
            cur_state = env.reset()
            rewards = [0.0]
            visited_states = []
            while not done:
                visited_states.append(cur_state)
                action = policy.step(cur_state)
                new_st, reward, done, *_ = env.step(action)
                rewards.append(reward)
                cur_state = new_st
            for i, state in enumerate(visited_states):
                if i + 1 >= len(rewards):
                    break
                discounted_return_from_state = \
                    np.dot(np.array(rewards[i + 1:]),
                           np.fromfunction(lambda i: discount ** i, ((len(rewards) - i - 1),)))
                state_values[state] += \
                    learning_rate * (discounted_return_from_state - state_values[state])
            if print_every is not None and episode % print_every == 0:
                print('State-Value estimation:\n{}'.format(state_values))
        return state_values

    @staticmethod
    def approx_state_value_eval(env, policy,
                                discount=0.999, learning_rate=0.01,
                                n_iter=1000, print_every=None):
        state_values = LinearApproximator(lambda x: x, env.observation_space.shape[0])
        for episode in range(n_iter):
            done = False
            cur_state = env.reset()
            rewards = [0.0]
            visited_states = []
            while not done:
                visited_states.append(cur_state)
                action = policy.step(cur_state)
                new_st, reward, done, *_ = env.step(action)
                rewards.append(reward)
                cur_state = new_st
            for i, state in enumerate(visited_states):
                if i + 1 >= len(rewards):
                    break
                discounted_return_from_state = \
                    np.dot(np.array(rewards[i + 1:]),
                           np.fromfunction(lambda i: discount ** i, (len(rewards) - i - 1, )))
                update = (discounted_return_from_state
                           - state_values.state_value(state)) * state_values.grad(state)
                state_values.update_w(update, learning_rate)
            if print_every is not None and episode % print_every == 0:
                print('State-Value estimation:\n{}'.format(
                    [(s, state_values.state_value(s)) for s in visited_states[:10]]
                ))
        return state_values

    @staticmethod
    def action_value_eval(env, policy,
                          discount=0.999, learning_rate=0.01,
                          n_iter=1000, print_every=None):
        action_values = [[0.0 for _ in range(env.action_space.n)] for _ in range(env.state_space.n)]

        for episode in range(n_iter):
            done = False
            cur_state = env.reset()
            rewards = [0.0]
            visited_state_action_pairs = []
            while not done:
                action = policy.step(cur_state)
                visited_state_action_pairs.append((cur_state, action))
                new_st, reward, done, *_ = env.step(action)
                rewards.append(reward)
                cur_state = new_st
            for i, (state, action) in enumerate(visited_state_action_pairs):
                if i + 1 >= len(rewards):
                    break
                discounted_return_from_state = \
                    np.dot(np.array(rewards[i + 1:]),
                           np.fromfunction(lambda i: discount ** i, ((len(rewards) - i - 1),)))
                action_values[state][action] += \
                    learning_rate * (discounted_return_from_state - action_values[state][action])
            if print_every is not None and episode % print_every == 0:
                print('Action-Value estimation:\n{}'.format(action_values))
        return action_values


class TDZero:
    @staticmethod
    def state_value_eval(env, policy,
                         discount=0.999, learning_rate=0.01,
                         n_iter=1000, print_every=None):
        state_values = [0.0 for _ in range(env.state_space.n)]

        for episode in range(n_iter):
            done = False
            cur_state = env.reset()
            while not done:
                action = policy.step(cur_state)
                new_st, reward, done, *_ = env.step(action)
                state_values[cur_state] += \
                    learning_rate \
                    * (reward + discount * state_values[new_st] - state_values[cur_state])
                cur_state = new_st
            if print_every is not None and episode % print_every == 0:
                print('State-Value estimation:\n{}'.format(state_values))
        return state_values

    @staticmethod
    def approx_state_value_eval(env, policy,
                                discount=0.999, learning_rate=0.01, feature_extractor=lambda x: x,
                                n_iter=1000):
        state_values = LinearApproximator(feature_extractor, env.observation_space.shape[0])

        for episode in range(n_iter):
            done = False
            cur_state = env.reset()
            while not done:
                action = policy.step(cur_state)
                new_st, reward, done, *_ = env.step(action)
                update = (reward + discount * state_values.state_value(new_st) - state_values.state_value(cur_state)) \
                         * state_values.grad(cur_state)
                state_values.update_w(update, learning_rate)
                cur_state = new_st
        return state_values

    @staticmethod
    def action_value_eval(env, policy,
                          discount=0.999,
                          learning_rate=0.01,
                          n_iter=1000, print_every=None):
        action_values = [[0.0 for _ in range(env.action_space.n)] for _ in range(env.state_space.n)]
        for episode in range(n_iter):
            done = False
            cur_state = env.reset()
            action = policy.step(cur_state)
            while not done:
                new_st, reward, done, *_ = env.step(action)
                next_action = policy.step(new_st)
                action_values[cur_state][action] += \
                    learning_rate * (reward + action_values[new_st][next_action] - action_values[cur_state][action])
                cur_state = new_st
                action = next_action
            if print_every is not None and episode % print_every == 0:
                learning_rate /= 2.0
                print('Action-Value estimation:\n{}'.format(action_values))
        return action_values


class TDLambda:
    @staticmethod
    def action_value_eval(env, policy, lamb,
            discount=0.999,
            learning_rate=0.01,
            n_iter=1000,
            print_every=None):
        action_values = [[0.0 for _ in range(env.action_space.n)] for _ in range(env.state_space.n)]
        for episode in range(n_iter):
            eligibility = [[0.0 for _ in range(env.action_space.n)] for _ in range(env.state_space.n)]
            done = False
            cur_state = env.reset()
            action = policy.step(cur_state)
            while not done:
                for s in range(env.state_space.n):
                    for a in range(env.action_space.n):
                        eligibility[s][a] *= lamb * discount
                eligibility[cur_state][action] += 1
                new_st, reward, done, *_ = env.step(action)
                next_action = policy.step(new_st)
                error = reward + action_values[new_st][next_action] - action_values[cur_state][action]
                for s in range(env.state_space.n):
                    for a in range(env.action_space.n):
                        action_values[s][a] += learning_rate * eligibility[cur_state][action] * error
                cur_state, action = new_st, next_action
            if print_every != None and episode % print_every == 0:
                print('Action-Value estimation:\n{}'.format(action_values))
        return action_values

    @staticmethod
    def state_value_eval(env, policy, lamb,
            discount=0.999,
            learning_rate=0.01,
            n_iter=1000,
            print_every=None):
        state_values = [0.0 for _ in range(env.state_space.n)]
        for episode in range(n_iter):
            eligibility = [0.0 for _ in range(env.state_space.n)]
            done = False
            cur_state = env.reset()
            while not done:
                for s in range(env.state_space.n):
                    eligibility[s] *= lamb * discount
                eligibility[cur_state] += 1
                action = policy.step(cur_state)
                new_st, reward, done, *_ = env.step(action)
                error = reward + state_values[new_st] - state_values[cur_state]
                for s in range(env.state_space.n):
                    state_values[s] += learning_rate * eligibility[s] * error
                cur_state = new_st
            if print_every != None and episode % print_every == 0:
                print('Action-Value estimation:\n{}'.format(state_values))
        return state_values
