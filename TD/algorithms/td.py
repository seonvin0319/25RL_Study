import numpy as np
import gym


def sarsa(Q, state, action, reward, next_state, next_action, alpha, gamma):
    return alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

def q_learning(Q, state, action, reward, next_state, alpha, gamma):
    return alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])