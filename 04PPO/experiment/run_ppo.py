import argparse
import yaml
from pathlib import Path
from algorithms.ppo import PPO
from environment.gym_trainer import GymTrainer

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """Parsing command line arguments"""
    parser = argparse.ArgumentParser(description='PPO Training Arguments')
    parser.add_argument('--config', type=str, default='./setup/ppo_arg.yaml',
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
    parser.add_argument('--actor-lr', type=float, help='override actor learning rate')
    parser.add_argument('--critic-lr', type=float, help='override critic learning rate')
    parser.add_argument('--gamma', type=float, help='override discount factor')
    parser.add_argument('--lam', type=float, help='override gae lambda')
    parser.add_argument('--eps-clip', type=float, help='override clip range')
    parser.add_argument('--entropy-coef', type=float, help='override entropy coefficient')
    parser.add_argument('--value-coef', type=float, help='override value coefficient')
    parser.add_argument('--rollout-steps', type=int, help='override rollout steps')
    parser.add_argument('--batch-size', type=int, help='override batch size')
    parser.add_argument('--update-epochs', type=int, help='override number of update epochs')
    parser.add_argument('--max-episodes', type=int, help='override max episodes')
    parser.add_argument('--max-steps', type=int, help='override max steps per episode')

    # save/load
    parser.add_argument('--save-model', action='store_true', help='save model')
    parser.add_argument('--model-path', type=str, help='override model path')
    parser.add_argument('--make-csv', action='store_true', help='save csv')
    parser.add_argument('--csv-path', type=str, help='override csv path')
    parser.add_argument('--load-model', type=str, help='override load model path')

    return parser.parse_args()

def update_config(config, args):
    """Update configuration dictionary with command line arguments"""

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
    train_args = ['actor_lr', 'critic_lr', 'gamma', 'lam', 'eps_clip',
                  'entropy_coef', 'value_coef', 'rollout_steps',
                  'batch_size', 'update_epochs', 'max_episodes', 'max_steps']

    for arg in train_args:
        val = getattr(args, arg.replace('-', '_'))
        if val is not None:
            config['train'][arg] = val

    # save/load
    if args.save_model:
        config['save']['model'] = True
    if args.model_path:
        config['save']['model_path'] = args.model_path
    if args.make_csv:
        config['save']['make_csv'] = True
    if args.csv_path:
        config['save']['csv_path'] = args.csv_path
    if args.load_model:
        config['save']['load_model'] = args.load_model

    return config

def main():
    args = parse_args()
    config = load_config(args.config)
    config = update_config(config, args)

    # object1 = GymTrainer instance
    trainer = GymTrainer(
        env_name=config['env']['name'],
        render_mode=config['env']['render_mode']
    )

    # object2 = Environment info
    state_dim, action_dim = trainer.get_env_info()

    # object3 = PPO agent instance
    ppo = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['model']['hidden_dim'],
        actor_lr=config['train']['actor_lr'],
        critic_lr=config['train']['critic_lr'],
        gamma=config['train']['gamma'],
        lam=config['train']['lam'],
        eps_clip=config['train']['eps_clip'],
        entropy_coef=config['train']['entropy_coef'],
        value_coef=config['train']['value_coef'],
        rollout_steps=config['train']['rollout_steps'],
        device=config['model']['device']
    )

    # logic1 = train PPO agent
    trainer.train(
        agent=ppo,
        max_episode_num=config['train']['max_episodes'],
        max_episode_length=config['train']['max_steps'],
        batch_size=config['train']['batch_size'],
        update_epochs=config['train']['update_epochs'],
        make_csv=config['save']['make_csv'],
        csv_dir=config['save']['csv_path'],
        save_model=config['save']['model'],
        model_dir=config['save']['model_path'],
        load_model_path=config['save']['load_model']
    )

    # logic2 = test PPO agent
    trainer.test(
        agent=ppo,
        max_episode_num=config['train']['max_episodes'],
        max_episode_length=config['train']['max_steps']
    )

if __name__ == '__main__':
    main()
