import numpy as np
import imageio
import gym

def policy_evaluation(policy, env, discount_factor, theta = 1e-6):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    value = np.zeros(state_size)
    while True:
        delta = 0
        for state in range(state_size):
            v = 0
            for action, action_prob in enumerate(policy[state]):
                for state_prob, next_state, reward, term in env.P[state][action]:
                    v += action_prob * state_prob * (reward + discount_factor * value[next_state])
            delta = max(delta, np.abs(value[state] - v))
            value[state] = v
        if delta < theta:
            print(f'policy evaluation : {value}')
            return value

def one_step_lookahead(env, state, value, discount_factor):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    action_value = np.zeros(action_size)
    for action in range(action_size):
        for prob, next_state, reward, term in env.P[state][action]:
            action_value[action] += prob * (reward + discount_factor * value[next_state])
    return action_value

def policy_iteration(env, discount_factor = 1.0):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    policy = np.ones((state_size, action_size)) / action_size

    evaluated_policies = 1
    while True:
        stable_policy = True
        value = policy_evaluation(policy, env, discount_factor = discount_factor)
        for state in range(state_size):
            current_action = np.argmax(policy[state])
            action_value = one_step_lookahead(env, state, value, discount_factor)
            best_action = np.argmax(action_value)
            if current_action != best_action:
                stable_policy = True
                policy[state] = np.eye(action_size)[best_action]
        if stable_policy:
            print(f"evaluated policies : {policy}")
            return policy, value