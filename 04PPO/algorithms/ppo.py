import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64,
                 actor_lr=3e-4, critic_lr=1e-3, gamma=0.99, lam=0.95,
                 eps_clip=0.2, entropy_coef=0.01, value_coef=0.5,
                 rollout_steps=2048, device='cpu'):
        """
        object1 = actor network (policy network)
        object2 = critic network (value network)
        object3 = trajectory buffer
        object4 = optimizers (actor, critic)

        hyperparameter
        - hidden_dim : hidden layer size
        - actor_lr : learning rate for actor network
        - critic_lr : learning rate for critic network
        - gamma : discount factor
        - lam : GAE lambda (for advantage calculation)
        - eps_clip : PPO clip range (for stable updates)
        - entropy_coef : coefficient for entropy bonus (exploration)
        - value_coef : coefficient for critic loss
        - rollout_steps : number of steps per trajectory rollout
        - device : cpu or cuda
        """
        super(PPO, self).__init__()

        self.device = device

        # hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.rollout_steps = rollout_steps

        # object1 = actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        ).to(self.device)

        # object2 = critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        # log_std parameter (for continuous action distribution)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # object4 = optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # object3 = trajectory buffer
        self.buffer = []

    def sample_action(self, state):
        """
        logic1 = sample an action and log probability from policy
        """
        state = torch.FloatTensor(state).to(self.device)
        mean = self.actor(state)
        std = self.log_std.exp()
        dist = Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        return action.cpu().detach().numpy(), log_prob.item()

    def store_transition(self, transition):
        """
        logic2 = store one transition (s, a, r, s', done, log_prob)
        """
        self.buffer.append(transition)

    def compute_gae(self, rewards, values, dones, next_value):
        """
        logic3 = compute returns and advantages using GAE
        """
        gae = 0
        returns = []
        values = values + [next_value]

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])

        return returns

    def update(self, batch_size=64, update_epochs=10):
        """
        logic4 = update actor and critic network using PPO objective
        """

        # unpack buffer
        states, actions, rewards, next_states, dones, log_probs = zip(*self.buffer)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(log_probs)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)

        # compute returns and advantages
        with torch.no_grad():
            values = self.critic(states).squeeze().cpu().numpy().tolist()
            next_state = torch.FloatTensor(np.array(next_states[-1])).to(self.device)
            next_value = self.critic(next_state).item()

        returns = self.compute_gae(rewards, values, dones, next_value)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - self.critic(states).squeeze()

        dataset_size = states.size(0)

        # logic4-1 = multiple epochs of mini-batch updates
        for _ in range(update_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for i in range(0, dataset_size, batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # predict new log probs
                mean = self.actor(batch_states)
                std = self.log_std.exp()
                dist = Normal(mean, std)

                new_log_probs = dist.log_prob(batch_actions).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1).mean()

                # ratio for importance sampling
                ratio = (new_log_probs - batch_old_log_probs).exp()

                # surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic loss
                value_pred = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(value_pred, batch_returns)

                # total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                # optimize networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # logic4-2 = clear buffer after update
        self.buffer.clear()

    def save_model(self, path):
        """
        logic5 = save actor and critic parameters
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load_model(self, path):
        """
        logic6 = load actor and critic parameters
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

    def eval(self):
        """
        logic7 = set networks to evaluation mode
        """
        self.actor.eval()
        self.critic.eval()
