import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


def split_and_pad_trajectories(tensor: torch.Tensor, dones: torch.Tensor):
    """Split trajectories at done indices and pad to max length. Returns (padded, masks)."""
    dones = dones.clone()
    dones[-1] = 1
    # flatten dones in env-major order
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()

    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    trajectories = (*trajectories, torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device))
    padded = torch.nn.utils.rnn.pad_sequence(trajectories)
    padded = padded[:, :-1]

    masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded, masks


class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape, device):
        self.device = device

        # Core
        self.critic_obs = np.zeros([num_transitions_per_env, num_envs, *critic_obs_shape], dtype=np.float32)
        self.actor_obs = np.zeros([num_transitions_per_env, num_envs, *actor_obs_shape], dtype=np.float32)
        self.rewards = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.actions = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)
        self.dones = np.zeros([num_transitions_per_env, num_envs, 1], dtype=bool)

        # For PPO
        self.actions_log_prob = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.values = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.returns = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.advantages = np.zeros([num_transitions_per_env, num_envs, 1], dtype=np.float32)
        self.mu = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)
        self.sigma = np.zeros([num_transitions_per_env, num_envs, *actions_shape], dtype=np.float32)

        # torch variables
        self.critic_obs_tc = torch.from_numpy(self.critic_obs).to(self.device)
        self.actor_obs_tc = torch.from_numpy(self.actor_obs).to(self.device)
        self.actions_tc = torch.from_numpy(self.actions).to(self.device)
        self.actions_log_prob_tc = torch.from_numpy(self.actions_log_prob).to(self.device)
        self.values_tc = torch.from_numpy(self.values).to(self.device)
        self.returns_tc = torch.from_numpy(self.returns).to(self.device)
        self.advantages_tc = torch.from_numpy(self.advantages).to(self.device)
        self.mu_tc = torch.from_numpy(self.mu).to(self.device)
        self.sigma_tc = torch.from_numpy(self.sigma).to(self.device)

        # saved hidden states for recurrent policies
        self.saved_hidden_state_a = None
        self.saved_hidden_state_c = None

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.device = device

        self.step = 0

    def add_transitions(self, actor_obs, critic_obs, actions, mu, sigma, rewards, dones, actions_log_prob,
                        hidden_state_a=None, hidden_state_c=None):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.critic_obs[self.step] = critic_obs
        self.actor_obs[self.step] = actor_obs
        self.actions[self.step] = actions
        self.mu[self.step] = mu
        self.sigma[self.step] = sigma
        self.rewards[self.step] = rewards.reshape(-1, 1)
        self.dones[self.step] = dones.reshape(-1, 1)
        self.actions_log_prob[self.step] = actions_log_prob.reshape(-1, 1)
        self._save_hidden_states(hidden_state_a, hidden_state_c)
        self.step += 1

    def clear(self):
        self.step = 0

    def _save_hidden_states(self, hidden_state_a, hidden_state_c):
        if hidden_state_a is None and hidden_state_c is None:
            return

        if hidden_state_a is not None and not isinstance(hidden_state_a, tuple):
            hidden_state_a = (hidden_state_a,)
        if hidden_state_c is not None and not isinstance(hidden_state_c, tuple):
            hidden_state_c = (hidden_state_c,)

        if self.saved_hidden_state_a is None and hidden_state_a is not None:
            self.saved_hidden_state_a = [
                torch.zeros(self.actor_obs.shape[0], *h.shape, device=self.device) for h in hidden_state_a
            ]
        if self.saved_hidden_state_c is None and hidden_state_c is not None:
            self.saved_hidden_state_c = [
                torch.zeros(self.actor_obs.shape[0], *h.shape, device=self.device) for h in hidden_state_c
            ]

        if hidden_state_a is not None:
            for i in range(len(hidden_state_a)):
                self.saved_hidden_state_a[i][self.step].copy_(hidden_state_a[i])
        if hidden_state_c is not None:
            for i in range(len(hidden_state_c)):
                self.saved_hidden_state_c[i][self.step].copy_(hidden_state_c[i])

    def compute_returns(self, last_values, critic, gamma, lam):
        with torch.no_grad():
            if getattr(critic.architecture, "is_recurrent", False):
                values = []
                hidden = torch.zeros(1,
                                     self.num_envs,
                                     critic.architecture.hidden_size,
                                     device=self.device,
                                     dtype=torch.float32)
                for t in range(self.num_transitions_per_env):
                    done_mask = torch.from_numpy(self.dones[t].astype(float)).to(self.device).view(1, -1, 1)
                    if done_mask.dtype != hidden.dtype:
                        done_mask = done_mask.to(dtype=hidden.dtype)
                    hidden = hidden * (1.0 - done_mask)
                    obs_t = torch.from_numpy(self.critic_obs[t]).to(self.device, dtype=torch.float32)
                    if hidden.dtype != obs_t.dtype:
                        hidden = hidden.to(dtype=obs_t.dtype)
                    val_t, hidden = critic.predict_recurrent(obs_t, hidden)
                    values.append(val_t.cpu().numpy())
                self.values = np.stack(values, axis=0)
            else:
                self.values = critic.predict(torch.from_numpy(self.critic_obs).to(self.device)).cpu().numpy()

        advantage = 0

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values.cpu().numpy()
                # next_is_not_terminal = 1.0 - self.dones[step].float()
            else:
                next_values = self.values[step + 1]
                # next_is_not_terminal = 1.0 - self.dones[step+1].float()

            next_is_not_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        # Convert to torch variables
        self.critic_obs_tc = torch.from_numpy(self.critic_obs).to(self.device)
        self.actor_obs_tc = torch.from_numpy(self.actor_obs).to(self.device)
        self.actions_tc = torch.from_numpy(self.actions).to(self.device)
        self.actions_log_prob_tc = torch.from_numpy(self.actions_log_prob).to(self.device)
        self.values_tc = torch.from_numpy(self.values).to(self.device)
        self.returns_tc = torch.from_numpy(self.returns).to(self.device)
        self.advantages_tc = torch.from_numpy(self.advantages).to(self.device)
        self.sigma_tc = torch.from_numpy(self.sigma).to(self.device)
        self.mu_tc = torch.from_numpy(self.mu).to(self.device)

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        dones = torch.from_numpy(self.dones).to(self.device)

        padded_actor_obs, masks = split_and_pad_trajectories(self.actor_obs_tc, dones)
        padded_critic_obs, _ = split_and_pad_trajectories(self.critic_obs_tc, dones)
        padded_actions, _ = split_and_pad_trajectories(self.actions_tc, dones)
        padded_old_logp, _ = split_and_pad_trajectories(self.actions_log_prob_tc, dones)
        padded_adv, _ = split_and_pad_trajectories(self.advantages_tc, dones)
        padded_returns, _ = split_and_pad_trajectories(self.returns_tc, dones)
        padded_values, _ = split_and_pad_trajectories(self.values_tc, dones)

        num_mini_batches = max(1, min(num_mini_batches, self.num_envs))
        mini_batch_size = max(1, self.num_envs // num_mini_batches)
        for _ in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones_t = dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones_t, dtype=torch.bool)
                last_was_done[1:] = dones_t[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = masks[:, first_traj:last_traj]
                actor_obs_batch = padded_actor_obs[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs[:, first_traj:last_traj]
                actions_batch = padded_actions[:, first_traj:last_traj]
                old_logp_batch = padded_old_logp[:, first_traj:last_traj]
                advantages_batch = padded_adv[:, first_traj:last_traj]
                returns_batch = padded_returns[:, first_traj:last_traj]
                values_batch = padded_values[:, first_traj:last_traj]

                hidden_state_a_batch = None
                hidden_state_c_batch = None
                if self.saved_hidden_state_a is not None:
                    hidden_state_a_batch = [
                        saved_hidden_state.permute(2, 0, 1, 3)[last_was_done.transpose(1, 0)]
                        [first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_hidden_state_a
                    ]
                    hidden_state_a_batch = hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else hidden_state_a_batch
                if self.saved_hidden_state_c is not None:
                    hidden_state_c_batch = [
                        saved_hidden_state.permute(2, 0, 1, 3)[last_was_done.transpose(1, 0)]
                        [first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_hidden_state_c
                    ]
                    hidden_state_c_batch = hidden_state_c_batch[0] if len(hidden_state_c_batch) == 1 else hidden_state_c_batch

                yield (
                    actor_obs_batch,
                    critic_obs_batch,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_logp_batch,
                    (hidden_state_a_batch, hidden_state_c_batch),
                    masks_batch,
                )

                first_traj = last_traj

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
            actor_obs_batch = self.actor_obs_tc.view(-1, *self.actor_obs_tc.size()[2:])[indices]
            critic_obs_batch = self.critic_obs_tc.view(-1, *self.critic_obs_tc.size()[2:])[indices]
            actions_batch = self.actions_tc.view(-1, self.actions_tc.size(-1))[indices]
            sigma_batch = self.sigma_tc.view(-1, self.sigma_tc.size(-1))[indices]
            mu_batch = self.mu_tc.view(-1, self.mu_tc.size(-1))[indices]
            values_batch = self.values_tc.view(-1, 1)[indices]
            returns_batch = self.returns_tc.view(-1, 1)[indices]
            old_actions_log_prob_batch = self.actions_log_prob_tc.view(-1, 1)[indices]
            advantages_batch = self.advantages_tc.view(-1, 1)[indices]
            yield actor_obs_batch, critic_obs_batch, actions_batch, sigma_batch, mu_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for batch_id in range(num_mini_batches):
            yield self.actor_obs_tc.view(-1, *self.actor_obs_tc.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.critic_obs_tc.view(-1, *self.critic_obs_tc.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions_tc.view(-1, self.actions_tc.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.sigma_tc.view(-1, self.sigma_tc.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.mu_tc.view(-1, self.mu_tc.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.values_tc.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.advantages_tc.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.returns_tc.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions_log_prob_tc.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
