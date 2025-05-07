import gym
import csv
import numpy as np
from collections import deque
import os

class GymTrainer:
    def __init__(self, env_name, render_mode='human'):
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = None

    def get_env_info(self):
        # return the dimension of state and action
        env = gym.make(self.env_name)
        state_size = env.observation_space.shape[0]
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
        # initialize environment
        self.env = gym.make(self.env_name, render_mode=self.render_mode)
        state, _ = self.env.reset()
        return state

    def _save_to_csv(self, csv_dir, data):
        # save data to csv file
        with open(csv_dir, 'a') as f:
            writer = csv.writer(f)
            if isinstance(data, list):
                writer.writerow(data)
            else:
                writer.writerow([data])

    def train(self, agent, max_episode_num, max_episode_length,
              batch_size, make_csv=False, csv_dir=None,
              load_model_path=None, save_model=False, model_dir=None):

        if self.env is None:
            self._init_env()

        if load_model_path is not None:
            agent.load_model(load_model_path)

        # save data to csv file
        if make_csv:
            if csv_dir is None:
                raise ValueError("csv_dir must be specified.")
            if not csv_dir.endswith('.csv'):
                csv_dir += '.csv'
            os.makedirs(os.path.dirname(csv_dir), exist_ok=True)
            with open(csv_dir, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Train Loss'])

        recent_rewards = deque(maxlen=100)
        best_avg_reward = 0.0
        wins = 0
        total_reward = 0

        for episode in range(max_episode_num):
            state = self._init_env()
            terminated = False
            truncated = False
            done = False
            step = 0
            episode_reward = 0

            while not done:
                step += 1
                action = agent.sample_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if agent.replay_buffer is not None:
                    agent.replay_buffer.push(state, action, reward, next_state, done)

                train_loss = agent.update(batch_size)
                state = next_state
                episode_reward += reward

                if make_csv and train_loss is not None:
                    self._save_to_csv(csv_dir, train_loss)

                if step > max_episode_length:
                    done = True

                # if step % 50 == 0:
                #     print(f"[Episode {episode} | Step {step}]  Episode reward: {episode_reward:.2f}")

            if episode % 10 == 0:
                print(f"[Episode {episode}]  Total reward: {total_reward / (episode + 1):.4f}")

            if episode_reward >= 475:
                wins += 1
            total_reward += episode_reward
            recent_rewards.append(episode_reward)

            if make_csv:
                self._save_to_csv(csv_dir, [episode, episode_reward])

            if len(recent_rewards) == 100:
                avg_reward = sum(recent_rewards) / 100
                if save_model and avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    model_path = model_dir or './model.pth'
                    if not model_path.endswith('.pth'):
                        model_path += '.pth'
                    agent.save_model(model_path)
                    print(f"✅ Saved model at {model_path} (avg_reward = {avg_reward:.2f})")

            # if save_model and episode_reward >= 475:
            #     if model_dir is None:
            #         model_path = './model.pth'
            #     elif not model_dir.endswith('.pth'):
            #         model_path = model_dir + '.pth'
            #     else:
            #         model_path = model_dir
            #     agent.save_model(model_path)

        average_reward = total_reward / max_episode_num
        print(f'Number of Wins over {max_episode_num} episodes: {wins}')
        print(f'Average Reward: {average_reward:.4f}')

        self.env.close()
        return average_reward

    def test(self, agent, max_episode_num, max_episode_length):
        
        agent.eval()

        for episode in range(max_episode_num):
            state = self._init_env()
            episode_return = 0
            step = 0

            while True:
                step += 1
                action = agent.sample_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                state = next_state
                episode_return += reward

                if done or step >= max_episode_length:
                    break

            print(f'Episode {episode} Return: {episode_return} Steps: {step}')

        self.env.close()