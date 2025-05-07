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
        env = gym.make(self.env_name, is_slippery=False)
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
        self.env = gym.make(self.env_name, render_mode=self.render_mode, is_slippery=False)
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

    def epsilon_greedy(self, Q, epsilon):
        policy = np.ones((self.env.observation_space.n, self.env.action_space.n)) * epsilon / self.env.action_space.n
        policy[np.arange(self.env.observation_space.n), np.argmax(Q, axis=1)] += 1 - epsilon
        return policy
        
    def train(self, solver_name, solver_func, n_episodes, epsilon, alpha, gamma, make_csv=False, csv_dir=None):
        if self.env is None:
            self._init_env()
        state_size = self.env.observation_space.n
        action_size = self.env.action_space.n
        Q = np.zeros((state_size, action_size))

        if make_csv:
            if csv_dir is None:
                raise ValueError("csv_dir must be specified to distinguish between solvers.")
            if not csv_dir.endswith('.csv'):
                csv_dir += '.csv'
            os.makedirs(os.path.dirname(csv_dir), exist_ok=True)
            with open(csv_dir, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Reward'])

        wins = 0
        total_reward = 0
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            policy = self.epsilon_greedy(Q, epsilon)
            action = np.random.choice(action_size, p=policy[state])
            terminated = False
            truncated = False
            done = False
            step = 0
            while not done:
                step += 1
                policy = self.epsilon_greedy(Q, epsilon)
                if solver_name == "q_learning":
                    action = np.random.choice(action_size, p=policy[state])
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                if terminated and reward == 0.0:
                    reward = -1.0

                if solver_name == "sarsa":
                    next_action = np.random.choice(action_size, p=policy[next_state])
                    Q[state,action] += solver_func(Q, state, action, reward, next_state, next_action, alpha, gamma)
                else:
                    Q[state,action] += solver_func(Q, state, action, reward, next_state, alpha, gamma)

                state = next_state
                if solver_name == 'sarsa':
                    action = next_action

                if step > 100:
                    done = 1

            if episode % 10 == 0:
                print(f"[Episode {episode}] average reward: {total_reward / (episode + 1):.4f}")


            if done and reward == 1.0:
                wins += 1
                total_reward += reward

            if make_csv:
                self._save_to_csv(csv_dir, [episode, reward])

        average_reward = total_reward / n_episodes
        print(f'Number of Wins over {n_episodes} episodes: {wins}')
        print(f'Average Reward: {average_reward:.4f}')

        self.env.close()
        return policy

    def test(self, policy, n_episodes):
        self._init_env()
        wins = 0
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            terminated = False
            truncated = False
            done = False

            action = policy[state]
            while not done:
                action = policy[state]
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                if terminated:
                    if reward == 0.0:
                        reward = -1.0

            if done and reward == 1.0:
                wins += 1
                total_reward += reward
                break

        average_reward = total_reward / n_episodes
        print(f'Number of Wins over {n_episodes} episodes: {wins}')
        print(f'Average Reward: {average_reward:.4f}')

        print(f"성공률: {wins / n_episodes * 100:.2f}%")
        return wins