from env import DiscreteEnvironment
from policy import DiscretePolicy, ContinuousStatePolicy

import numpy as np
from gym.spaces import Box, Discrete

import gym

def simple_dynamics(state, action):
    if action == 0:
        new_state = state - 1
    else:
        new_state = state + 1
    return new_state


def rewards(state: int, action: int) -> int:
    return int(simple_dynamics(state, action) == 6)


def rewards_builder(rows, cols):
    return np.array(
        [rewards(*pair) for pair in zip(rows.ravel(), cols.ravel())]
    ).reshape(rows.shape)


env = DiscreteEnvironment.Builder()\
    .set_n_actions(2)\
    .set_n_states(7)\
    .set_transition_dynamics(simple_dynamics)\
    .set_rewards(np.fromfunction(rewards_builder, (7, 2)))\
    .set_terminal_states([0, 6])\
    .build()

policy = DiscretePolicy(Discrete(7), Discrete(2))

# for _ in range(5):
#     env.reset()
#     total_reward = 0.0
#     discount = 1.0
#     while not env.done:
#         action = policy.step(env.cur_state)
#         new_state, reward, done = env.step(action)
#         total_reward += discount * reward
#         discount *= 0.999
#         print(new_state, reward, done, end='; ')
#
#     print('Total reward for last episode: {:f}'.format(total_reward))

print('Evaluate using Monte Carlo!')
# from policy_evaluation import monte_carlo_eval
from policy_evaluation import MonteCarlo
state_val = MonteCarlo.state_value_eval(env, policy)
action_val = MonteCarlo.action_value_eval(env, policy)
print('Monte Carlo Results:')
print('State values:\n{}'.format(
    dict(enumerate(state_val))
))
print('Action Values:\n{}'.format(
    dict(enumerate(action_val))
))
env_2 = gym.make('CartPole-v0')
policy_2 = ContinuousStatePolicy(env_2.observation_space, env_2.action_space)
state_val = MonteCarlo.approx_state_value_eval(env_2, policy_2, print_every=100)
print('State Value calculated using Monte-Carlo weight vector:\n{}'.format(
    state_val.get_weight_vector()
))

from policy_evaluation import TDZero
state_val = TDZero.state_value_eval(env, policy)
action_val = TDZero.action_value_eval(env, policy)
print('TD(0) Results:')
print('State values:\n{}'.format(
    dict(enumerate(state_val))
))
print('Action Values:\n{}'.format(
    dict(enumerate(action_val))
))

from policy_evaluation import TDLambda
lamb = 0.1
state_val = TDLambda.state_value_eval(env, policy, lamb)
action_val = TDLambda.action_value_eval(env, policy, lamb)
print('TD(lambda={:.3f}) Results:'.format(lamb))
print('State values:\n{}'.format(
    dict(enumerate(state_val))
))
print('Action Values:\n{}'.format(
    dict(enumerate(action_val))
))