import numpy as np
import gym


def value_iteration(env, discount_factor = 1.0, theta = 1e-6):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    value = np.zeros(state_size)

    while True:
        delta = 0
        for state in range(state_size):
            action_value = np.zeros(action_size)
            for action in range(action_size):
                for prob, next_state, reward, term in env.P[state][action]:
                    action_value[action] += prob * (reward + discount_factor * value[next_state])
            
            # directly choose maximum action value!
            best_action = np.max(action_value) 
            delta = max(delta, np.abs(value[state] - best_action))
            value[state] = best_action

        if delta < theta:
            print(f'value iteration converged : {value}')
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([state_size, action_size])
    for state in range(state_size):
        action_value = np.zeros(action_size)
        for action in range(action_size):
            for prob, next_state, reward, term in env.P[state][action]:
                action_value[action] += prob * (reward + discount_factor * value[next_state])
        best_action = np.argmax(action_value)
        policy[state, best_action] = 1.0
    return policy, value


        
        