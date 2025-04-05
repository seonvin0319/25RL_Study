import argparse
import yaml
import os
import time
import numpy as np
from pathlib import Path
from algorithms.policy_iteration import policy_iteration
from algorithms.value_iteration import value_iteration
from environment.gym_trainer import GymTrainer

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description='Dynamic Programming: Policy vs Value Iteration')
    parser.add_argument('--config', type=str, default='./setup/dp_arg.yaml', help='path to config file')
    parser.add_argument('--env-name', type=str, help='override environment name')
    parser.add_argument('--render-mode', type=str, choices=[None, 'human', 'rgb_array'], help='render mode')
    parser.add_argument('--episodes', type=int, help='number of evaluation episodes')
    return parser.parse_args()

def update_config(config, args):
    if args.env_name:
        config['env']['name'] = args.env_name
    if args.render_mode:
        config['env']['render_mode'] = args.render_mode
    if args.episodes:
        config['planner']['episodes'] = args.episodes
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    config = update_config(config, args)

    env_name = config['env']['name']
    render_mode = config['env']['render_mode']
    n_episodes = config['planner']['episodes']

    solvers = [
        ('policy', policy_iteration),
        ('value', value_iteration)
    ]

    for solver_name, solver_func in solvers:
        print(f"\n=== Running {solver_name.title()} Iteration ===")
        trainer = GymTrainer(env_name=env_name, render_mode=render_mode)
        trainer.get_env_info()

        trainer._init_env()
        env = trainer.env
        start = time.time()
        policy, V = solver_func(env)
        end = time.time()

        print(f"{solver_name} iteration Time: {end - start:.4f} sec")

        if not os.path.exists("results"):
            os.mkdir("results")
        np.save(f"results/{solver_name}_iteration_policy.npy", policy)
        np.save(f"results/{solver_name}_iteration_value.npy", V)
        csv_path = f"results/{solver_name}_iteration_eval.csv"
        print(f"saved: results/{solver_name}_policy.npy & value.npy")

        trainer.evaluate_policy(
            policy=policy,
            n_episodes=n_episodes,
            make_csv=True,
            csv_dir=csv_path
        )

if __name__ == '__main__':
    main()