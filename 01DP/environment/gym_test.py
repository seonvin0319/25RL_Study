import gym
import time
import os
import numpy as np
import imageio
from gym.wrappers import RecordVideo


# env = gym.make('FrozenLake-v1', render_mode='human')
env = gym.make('FrozenLake-v1', render_mode='rgb_array')
# env = RecordVideo(env, video_folder='videos', episode_trigger=lambda e: True)
state_dim = env.observation_space.n
action_dim = env.action_space.n

solvers = [
        ('policy', None),
        ('value', None)
    ]

for solver_name, _ in solvers:
    print(f"\n=== Running {solver_name.title()} Iteration ===")
    policy = np.load(f'results/{solver_name}_iteration_policy.npy')
    value = np.load(f'results/{solver_name}_iteration_value.npy')

    if not os.path.exists("gif"):
        os.makedirs("gif")

    wins = 0
    total_reward = 0
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        state, _ = env.reset()
        frames = []
        terminated = False
        done = False
        step = 0
        while True:
            frame = env.render()
            frames.append(frame)
            action = np.argmax(policy[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            step += 1

            if step > 60:
                break

            if done and reward == 1.0:
                wins += 1
                total_reward += reward
                frame = env.render()
                frames.append(frame)
                break

    
        filename = f"gif/test_gif/{solver_name}_iteration_episode_{episode + 1}.gif"
        imageio.mimsave(filename, frames, fps = 5)
        print(f"saved {filename} with {len(frames)} frames")

    average_reward = total_reward / 5
    print(f'Number of Wins over 5 episodes: {wins}')
    print(f'Average Reward: {average_reward:.4f}')

env.close()