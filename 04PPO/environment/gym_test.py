import gym
import torch
import time
import os
import imageio
from algorithms.ppo import PPO
from collections import deque

# object1 = environment initialization
env = gym.make('HalfCheetah-v4', render_mode='rgb_array')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# object2 = PPO agent
agent = PPO(state_dim=state_dim, action_dim=action_dim)
agent.load_model('results/ppo_model.pth')
agent.eval()

num_episodes = 10
max_steps = 1000

recent_rewards = deque(maxlen=100)

# create directory for gifs
if not os.path.exists("gif"):
    os.makedirs("gif")

for episode in range(num_episodes):
    print(f"\nEpisode {episode + 1}")
    state, _ = env.reset()
    frames = []
    episode_reward = 0

    for t in range(max_steps):
        frame = env.render()
        frames.append(frame)

        action, _ = agent.sample_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        done = terminated or truncated
        if done:
            print(f"Finished in {t + 1} steps")
            break

        time.sleep(0.01)  # make animation smoother

    recent_rewards.append(episode_reward)

    filename = f"gif/ppo_episode_{episode + 1}.gif"
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved {filename} with {len(frames)} frames")

    print(f"Episode {episode + 1} Reward: {episode_reward:.2f}")

    if len(recent_rewards) == 100:
        avg_reward = sum(recent_rewards) / 100
        print(f"\nAverage Reward over last 100 episodes: {avg_reward:.2f}")

env.close()
