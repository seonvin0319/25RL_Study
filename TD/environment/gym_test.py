import gym
import time
import os
import numpy as np
import imageio
from gym.wrappers import RecordVideo


# env = gym.make('FrozenLake-v1', render_mode='human')
env = gym.make('FrozenLake-v1', render_mode='rgb_array', is_slippery=False)
# env = RecordVideo(env, video_folder='videos', episode_trigger=lambda e: True)
state_dim = env.observation_space.n
action_dim = env.action_space.n

solvers = [
        ('sarsa'),
        ('q_learning')
    ]

for solver_name in solvers:
    print(f"\n=== Running {solver_name.title()} Iteration ===")
    policy = np.load(f'results/{solver_name}_policy.npy')

    if not os.path.exists("gif"):
        os.makedirs("gif")

    wins = 0
    state, _ = env.reset()
    terminated = False
    truncated = False
    done = False
    step = 0
    frames = []

    while True:
        frame = env.render()
        frames.append(frame)
        if isinstance(policy[state], (np.ndarray, list)):
            action = int(np.argmax(policy[state]))
        else:
            action = int(policy[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if terminated:
            if reward == 0.0:
                reward = -1.0
        step += 1

        if step > 60:
            break

        if done and reward == 1.0:
            wins += 1
            frame = env.render()
            frames.append(frame)
            break


    filename = f"gif/{solver_name}.gif"
    imageio.mimsave(filename, frames, fps=5)
    print(f"saved {filename} with {len(frames)} frames")

    print(f'Win or not: {wins}')
    print(f'Reward: {reward}')

    print(f"{solver_name} 성공률: {wins * 100:.2f}%")


env.close()