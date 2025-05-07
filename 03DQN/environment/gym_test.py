import gym
import torch
import time
import os
import imageio
from algorithms.dqn import DQN
from gym.wrappers import RecordVideo
from collections import deque

# env = gym.make('CartPole-v1', render_mode='human')
env = gym.make('CartPole-v1', render_mode='rgb_array')
# env = RecordVideo(env, video_folder='videos', episode_trigger=lambda e: True)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model_path = "results/model.pth"

agent = DQN(state_dim, action_dim)
agent.load_model(model_path)
agent.eval()

num_episodes = 100
max_steps = 800

recent_rewards = deque(maxlen=100)

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

        action = agent.sample_action(state)
        state, reward, term, trun, info = env.step(action)
        episode_reward += reward

        if term or trun:
            print(f"Finished in {t + 1} steps")
            break

        time.sleep(0.1)

    recent_rewards.append(episode_reward)

    if episode_reward >= 475:
        result = "✅ Episode Success"
    else:
        result = "❌ Episode Failure"

    print(f"Finished in {t + 1} steps - {result} (Reward: {episode_reward:.1f})")

    filename = f"gif/episode_{episode + 1}.gif"
    imageio.mimsave(filename, frames, fps=30)
    print(f"saved {filename} with {len(frames)} frames")

    if len(recent_rewards) == 100:
        avg_reward = sum(recent_rewards) / 100
        if avg_reward >= 475:
            print(f"\n🎉 Environment Solved! Average Reward over 100 episodes: {avg_reward:.2f}")
        else:
            print(f"Average Reward over last 100 episodes: {avg_reward:.2f}")

env.close()