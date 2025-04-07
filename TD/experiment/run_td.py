import argparse
import yaml
import os
import time
import numpy as np
from pathlib import Path
from algorithms.td import sarsa
from algorithms.td import q_learning
from environment.gym_trainer import GymTrainer

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description='Temporal Difference: SARSA vs Q-learning')
    parser.add_argument('--config', type=str, default='./setup/td_arg.yaml', help='path to config file')
    parser.add_argument('--env-name', type=str, help='override environment name')
    parser.add_argument('--render-mode', type=str, choices=[None, 'human', 'rgb_array'], help='render mode')
    parser.add_argument('--is_slippery', type=float, help='override whether env is slippery')
    parser.add_argument('--episodes', type=int, help='number of evaluation episodes')
    parser.add_argument('--test_episodes', type=int, help='number of evaluation test episodes')
    parser.add_argument('--gamma', type=float, help='override discount factor')
    parser.add_argument('--alpha', type=float, help='override alpha ratio')
    parser.add_argument('--epsilon', type=float, help='override initial epsilon')

    return parser.parse_args()

def update_config(config, args):
    if args.env_name:
        config['env']['name'] = args.env_name
    if args.render_mode:
        config['env']['render_mode'] = args.render_mode
    if args.is_slippery:
        config['env']['is_slippery'] = args.is_slippery
    if args.episodes:
        config['train']['episodes'] = args.episodes
    if args.test_episodes:
        config['test']['episodes'] = args.test_episodes
    if args.gamma:
        config['hyperparameter']['gamma'] = args.gamma
    if args.alpha:
        config['hyperparameter']['alpha'] = args.alpha
    if args.epsilon:
        config['hyperparameter']['epsilon'] = args.epsilon
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    config = update_config(config, args)

    env_name = config['env']['name']
    render_mode = config['env']['render_mode']
    is_slippery = config['env']['is_slippery']
    n_episodes = config['train']['episodes']
    test_episodes = config['test']['episodes']
    gamma = config['hyperparameter']['gamma']
    alpha = config['hyperparameter']['alpha']
    epsilon = config['hyperparameter']['epsilon']

    solvers = [
        ('sarsa', sarsa),
        ('q_learning', q_learning)
    ]

    for solver_name, solver_func in solvers:
        print(f"\n=== Running {solver_name.title()} Iteration ===")
        trainer = GymTrainer(env_name=env_name, render_mode=render_mode)
        trainer.get_env_info()

        trainer._init_env()
        env = trainer.env

        csv_path = f"results/{solver_name}_eval.csv"
        policy = trainer.train(
            solver_name,
            solver_func,
            n_episodes=n_episodes,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            make_csv=True,
            csv_dir=csv_path
        )

        if not os.path.exists("results"):
            os.mkdir("results")
        np.save(f"results/{solver_name}_policy.npy", policy)
        print(f"saved: results/{solver_name}_policy.npy")

if __name__ == '__main__':
    main()