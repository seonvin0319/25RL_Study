import argparse
import yaml
from pathlib import Path
from algorithms.dqn import DQN
from environment.gym_trainer import GymTrainer

def load_config(config_path):
    """load YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """parsing command line arguments"""
    parser = argparse.ArgumentParser(description='DQN Training Arguments')
    parser.add_argument('--config', type=str, default='./setup/dqn_arg.yaml',
                        help='path to config file')

    # environment
    parser.add_argument('--env-name', type=str, help='override environment name')
    parser.add_argument('--render-mode', type=str, choices=['human', 'rgb_array'],
                        help='override render mode')

    # model
    parser.add_argument('--hidden-dim', type=int, help='override hidden dimension')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='override device')

    # train
    parser.add_argument('--lr', type=float, help='override learning rate')
    parser.add_argument('--gamma', type=float, help='override discount factor')
    parser.add_argument('--epsilon', type=float, help='override initial epsilon')
    parser.add_argument('--epsilon-min', type=float, help='override minimum epsilon')
    parser.add_argument('--epsilon-decay', type=float, help='override epsilon decay rate')
    parser.add_argument('--buffer-size', type=int, help='override buffer size')
    parser.add_argument('--update-freq', type=int, help='override update frequency')
    parser.add_argument('--target-hard-update', action='store_true', help='use hard update')
    parser.add_argument('--batch-size', type=int, help='override batch size')
    parser.add_argument('--max-episodes', type=int, help='override max episodes')
    parser.add_argument('--max-steps', type=int, help='override max steps')

    # save/load
    parser.add_argument('--save-model', action='store_true', help='save model')
    parser.add_argument('--model-path', type=str, help='override model path')
    parser.add_argument('--make-csv', action='store_true', help='save csv')
    parser.add_argument('--csv-path', type=str, help='override csv path')
    parser.add_argument('--load-model', type=str, help='override load model path')

    return parser.parse_args()

def update_config(config, args):
    """update setting as command line arguments"""
    # environment
    if args.env_name:
        config['env']['name'] = args.env_name
    if args.render_mode:
        config['env']['render_mode'] = args.render_mode

    # model
    if args.hidden_dim:
        config['model']['hidden_dim'] = args.hidden_dim
    if args.device:
        config['model']['device'] = args.device

    # train
    train_args = ['lr', 'gamma', 'epsilon', 'epsilon_min', 'epsilon_decay',
                 'buffer_size', 'update_freq', 'batch_size', 'max_episodes', 'max_steps']

    for arg in train_args:
        val = getattr(args, arg.replace('-', '_'))
        if val is not None:
            config['train'][arg] = val

    if args.target_hard_update:
        config['train']['target_hard_update'] = True

    # save/load
    if args.save_model:
        config['save']['model'] = False
    if args.model_path:
        config['save']['model_path'] = args.model_path
    if args.make_csv:
        config['save']['make_csv'] = False
    if args.csv_path:
        config['save']['csv_path'] = args.csv_path
    if args.load_model:
        config['save']['load_model'] = args.load_model

    return config

def main():
    args = parse_args()
    config = load_config(args.config)
    config = update_config(config, args)

    # instant GymTrainer
    trainer = GymTrainer(
        env_name=config['env']['name'],
        render_mode=config['env']['render_mode']
    )

    # environment informantion
    state_dim, action_dim = trainer.get_env_info()

    # instant DQN agent
    dqn = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['model']['hidden_dim'],
        device=config['model']['device'],
        lr=config['train']['lr'],
        gamma=config['train']['gamma'],
        eps=config['train']['epsilon'],
        eps_min=config['train']['epsilon_min'],
        eps_decay=config['train']['epsilon_decay'],
        buffer_capacity=config['train']['buffer_size'],
        update_frequency=config['train']['update_freq'],
        target_net_hard_update=config['train']['target_hard_update']
    )

    # train
    trainer.train(
        agent=dqn,
        max_episode_num=config['train']['max_episodes'],
        max_episode_length=config['train']['max_steps'],
        batch_size=config['train']['batch_size'],
        make_csv=config['save']['make_csv'],
        csv_dir=config['save']['csv_path'],
        save_model=config['save']['model'],
        model_dir=config['save']['model_path'],
        load_model_path=config['save']['load_model']
    )

    # test
    trainer.test(
        agent=dqn,
        max_episode_num=config['train']['max_episodes'],
        max_episode_length=config['train']['max_steps']
    )

if __name__ == '__main__':
    main()