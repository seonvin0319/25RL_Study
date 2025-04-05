import gym
import csv
import numpy as np
import os

class GymTrainer:
    def __init__(self, env_name, render_mode='human'):
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = None

    def get_env_info(self):
        """환경의 state와 action 차원을 반환"""
        env = gym.make(self.env_name)
        state_size = env.observation_space.n
        action_size = env.action_space.n
        print('-'*30)
        print('Environment Info')
        print(f'Environment: {self.env_name}')
        print(f'State Dimension: {state_size}')
        print(f'Action Dimension: {action_size}')
        print('-'*30)
        env.close()
        return state_size, action_size

    def _init_env(self):
        """환경 초기화"""
        self.env = gym.make(self.env_name, render_mode=self.render_mode)
        state, _ = self.env.reset()
        return state

    def _save_to_csv(self, csv_dir, data):
        """CSV 파일에 데이터 저장"""
        with open(csv_dir, 'a') as f:
            writer = csv.writer(f)
            if isinstance(data, list):
                writer.writerow(data)
            else:
                writer.writerow([data])

    def evaluate_policy(self, policy, n_episodes, make_csv=False, csv_dir=None):
        if self.env is None:
            self._init_env()

        wins = 0
        total_reward = 0

        if make_csv:
            if csv_dir is None:
                raise ValueError("csv_dir must be specified to distinguish between solvers.")
            if not csv_dir.endswith('.csv'):
                csv_dir += '.csv'
            os.makedirs(os.path.dirname(csv_dir), exist_ok=True)
            with open(csv_dir, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Reward'])

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            terminated = False
            done = False

            while not done:
                action = np.argmax(policy[state])
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                state = next_state

            total_reward += reward
            if done and reward == 1.0:
                wins += 1

            if make_csv:
                self._save_to_csv(csv_dir, [episode, reward])

        average_reward = total_reward / n_episodes
        print(f'Number of Wins over {n_episodes} episodes: {wins}')
        print(f'Average Reward: {average_reward:.4f}')

        self.env.close()
        return wins, total_reward, average_reward