from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .storage import RolloutStorage


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 learning_rate=5e-4,
                 max_grad_norm=0.5,
                 learning_rate_schedule='adaptive',
                 desired_kl=0.01,
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 shuffle_batch=True):

        # PPO components
        self.actor = actor
        self.critic = critic
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape, actor.action_shape, device)

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

        # ADAM
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl
        self.schedule = learning_rate_schedule

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None
        self.is_recurrent = getattr(self.actor.architecture, "is_recurrent", False)
        if self.is_recurrent:
            hidden_size = self.actor.architecture.hidden_size
            self.actor_hidden = torch.zeros(1, num_envs, hidden_size, device=self.device, dtype=torch.float32)
            self.critic_hidden = torch.zeros(1, num_envs, hidden_size, device=self.device, dtype=torch.float32)

    def act(self, actor_obs):
        self.actor_obs = actor_obs
        with torch.no_grad():
            obs_tc = torch.from_numpy(actor_obs).to(self.device, dtype=torch.float32)
            if self.is_recurrent:
                if self.actor_hidden.dtype != obs_tc.dtype:
                    self.actor_hidden = self.actor_hidden.to(dtype=obs_tc.dtype)
                self.actions, self.actions_log_prob, self.actor_hidden = self.actor.sample_recurrent(obs_tc, self.actor_hidden)
                self.actor_hidden = self.actor_hidden.detach()
            else:
                self.actions, self.actions_log_prob = self.actor.sample(obs_tc)
        return self.actions

    def step(self, value_obs, rews, dones):
        hidden_a = None
        hidden_c = None
        if self.is_recurrent:
            value_obs_tc = torch.from_numpy(value_obs).to(self.device, dtype=torch.float32)
            if self.critic_hidden.dtype != value_obs_tc.dtype:
                self.critic_hidden = self.critic_hidden.to(dtype=value_obs_tc.dtype)
            values, self.critic_hidden = self.critic.evaluate_recurrent(value_obs_tc, self.critic_hidden)
            self.critic_hidden = self.critic_hidden.detach()
            hidden_a = self.actor_hidden.detach()
            hidden_c = self.critic_hidden.detach()
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, self.actor.action_mean,
                                     self.actor.distribution.std_np, rews, dones,
                                     self.actions_log_prob, hidden_a, hidden_c)
        if self.is_recurrent:
            done_mask = torch.from_numpy(dones.astype(float)).to(self.device).view(1, -1, 1)
            self.actor_hidden = self.actor_hidden * (1.0 - done_mask)
            self.critic_hidden = self.critic_hidden * (1.0 - done_mask)

    def update(self, actor_obs, value_obs, log_this_iteration, update):
        if self.is_recurrent:
            last_values = self._predict_recurrent_last_values(value_obs)
        else:
            last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device, dtype=torch.float32))

        # Learning step
        self.storage.compute_returns(last_values.to(self.device), self.critic, self.gamma, self.lam)
        if self.is_recurrent:
            mean_value_loss, mean_surrogate_loss, infos = self._train_step_recurrent(log_this_iteration)
        else:
            mean_value_loss, mean_surrogate_loss, infos = self._train_step(log_this_iteration)
        self.storage.clear()

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

    def log(self, variables):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        mean_std = self.actor.distribution.std.mean()
        self.writer.add_scalar('PPO/value_function', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar('PPO/surrogate', variables['mean_surrogate_loss'], variables['it'])
        self.writer.add_scalar('PPO/mean_noise_std', mean_std.item(), variables['it'])
        self.writer.add_scalar('PPO/learning_rate', self.learning_rate, variables['it'])

    def _train_step(self, log_this_iteration):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, critic_obs_batch, actions_batch, old_sigma_batch, old_mu_batch, current_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.batch_sampler(self.num_mini_batches):

                actions_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch)
                value_batch = self.critic.evaluate(critic_obs_batch)

                # Adjusting the learning rate using KL divergence
                mu_batch = self.actor.action_mean
                sigma_batch = self.actor.distribution.std

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.no_grad():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.2)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.2)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = current_values_batch + (value_batch - current_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
                self.optimizer.step()

                if log_this_iteration:
                    mean_value_loss += value_loss.item()
                    mean_surrogate_loss += surrogate_loss.item()

        if log_this_iteration:
            num_updates = self.num_learning_epochs * self.num_mini_batches
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss, locals()

    def _predict_recurrent_last_values(self, value_obs):
        obs_tc = torch.from_numpy(value_obs).to(self.device)
        hidden = torch.zeros(1, self.num_envs, self.critic.architecture.hidden_size, device=self.device)
        values, _ = self.critic.predict_recurrent(obs_tc, hidden)
        return values

    def _train_step_recurrent(self, log_this_iteration):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (actor_obs_batch,
             critic_obs_batch,
             actions_batch,
             values_batch,
             advantages_batch,
             returns_batch,
             old_actions_log_prob_batch,
             hidden_states_batch,
             masks_batch) in generator:

            hidden_a, hidden_c = hidden_states_batch
            logits_seq, _ = self.actor.architecture.architecture(actor_obs_batch, hidden_a)
            values_seq, _ = self.critic.architecture.architecture(critic_obs_batch, hidden_c)

            mask_flat = masks_batch.reshape(-1)
            if not torch.any(mask_flat):
                continue

            act_dim = actions_batch.shape[-1]
            logits_flat = logits_seq.reshape(-1, act_dim)[mask_flat]
            actions_flat = actions_batch.reshape(-1, act_dim)[mask_flat]
            old_logp_flat = old_actions_log_prob_batch.reshape(-1)[mask_flat]
            adv_flat = advantages_batch.reshape(-1)[mask_flat]
            returns_flat = returns_batch.reshape(-1)[mask_flat]
            current_values_flat = values_batch.reshape(-1)[mask_flat]
            values_flat = values_seq.reshape(-1)[mask_flat]

            logp_flat, entropy_flat = self.actor.distribution.evaluate(logits_flat, actions_flat)

            ratio = torch.exp(logp_flat - old_logp_flat)
            surrogate = -adv_flat * ratio
            surrogate_clipped = -adv_flat * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = current_values_flat + (values_flat - current_values_flat).clamp(-self.clip_param, self.clip_param)
                value_losses = (values_flat - returns_flat).pow(2)
                value_losses_clipped = (value_clipped - returns_flat).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_flat - values_flat).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_flat.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
            self.optimizer.step()

            if log_this_iteration:
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        if log_this_iteration:
            mean_value_loss /= self.num_learning_epochs
            mean_surrogate_loss /= self.num_learning_epochs

        return mean_value_loss, mean_surrogate_loss, locals()
