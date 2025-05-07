import gym
import csv
import numpy as np
import os

class GymTrainer:
    def __init__(self, env_name, render_mode='human'):
        """
        object1 = gym environment manager
        """
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = None

    def get_env_info(self):
        """
        logic1 = return dimension of state and action
        """
        env = gym.make(self.env_name)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        print('-'*30)
        print('Environment Info')
        print(f'Environment: {self.env_name}')
        print(f'State Dimension: {state_size}')
        print(f'Action Dimension: {action_size}')
        print('-'*30)
        env.close()
        return state_size, action_size

    def _init_env(self):
        """
        logic2 = initialize gym environment
        """
        self.env = gym.make(self.env_name, render_mode=self.render_mode)
        state, _ = self.env.reset()
        return state

    def _save_to_csv(self, csv_dir, data):
        """
        logic3 = save data to csv file
        """
        with open(csv_dir, 'a') as f:
            writer = csv.writer(f)
            if isinstance(data, list):
                writer.writerow(data)
            else:
                writer.writerow([data])

    def train(self, agent, max_episode_num, max_episode_length,
              batch_size, update_epochs=10,
              make_csv=False, csv_dir=None,
              load_model_path=None, save_model=False, model_dir=None):
        """
        logic4 = main training loop
        """

        if self.env is None:
            self._init_env()

        if load_model_path is not None:
            agent.load_model(load_model_path)

        # save csv file
        if make_csv:
            if csv_dir is None:
                raise ValueError("csv_dir must be specified.")
            if not csv_dir.endswith('.csv'):
                csv_dir += '.csv'
            os.makedirs(os.path.dirname(csv_dir), exist_ok=True)
            with open(csv_dir, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Reward'])

        wins = 0
        total_reward = 0

        for episode in range(max_episode_num):
            state, _ = self.env.reset()
            terminated = False
            truncated = False
            done = False
            step = 0
            episode_reward = 0

            agent.buffer = []  # reset buffer for PPO

            while not done:
                step += 1
                action, log_prob = agent.sample_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # store (state, action, reward, next_state, done, log_prob)
                agent.store_transition((state, action, reward, next_state, done, log_prob))

                state = next_state
                episode_reward += reward

                # if rollout steps collected, then update
                if len(agent.buffer) >= agent.rollout_steps:
                    agent.update(batch_size=batch_size, update_epochs=update_epochs)
                    agent.buffer = []  # clear after update

                if step > max_episode_length:
                    done = True

                if step % 10 == 0:
                    print(f"[Episode {episode} | Step {step}]  Episode Reward: {episode_reward:.2f}")

            if make_csv:
                self._save_to_csv(csv_dir, [episode, episode_reward])

            if save_model:
                if model_dir is None:
                    model_path = './model.pth'
                elif not model_dir.endswith('.pth'):
                    model_path = model_dir + '.pth'
                else:
                    model_path = model_dir
                agent.save_model(model_path)

            if episode % 10 == 0:
                print(f"[Episode {episode}]  Average Reward: {total_reward / (episode + 1):.4f}")

            total_reward += episode_reward

        average_reward = total_reward / max_episode_num
        print(f'Average Reward over {max_episode_num} episodes: {average_reward:.4f}')

        self.env.close()
        return average_reward

    def test(self, agent, max_episode_num, max_episode_length):
        """
        logic5 = evaluation loop
        """
        state = self._init_env()
        agent.eval()

        for episode in range(max_episode_num):
            episode_return = 0
            step = 0

            while True:
                step += 1
                action, _ = agent.sample_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                episode_return += reward

                if done or step >= max_episode_length:
                    break

            print(f'Episode {episode} Return: {episode_return} Steps: {step}')

        self.env.close()
