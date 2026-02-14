import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal


class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device
        self.action_mean = None
        self.scripted_policy = None

    def sample(self, obs):
        if self.scripted_policy is not None:
            self.action_mean = self.scripted_policy(obs).cpu().numpy()
            actions, log_prob = self.distribution.sample(self.action_mean)
            return actions, log_prob
        if getattr(self.architecture, "is_recurrent", False):
            raise RuntimeError("Recurrent policy requires sample_recurrent()")
        self.action_mean = self.architecture.architecture(obs).cpu().numpy()
        actions, log_prob = self.distribution.sample(self.action_mean)
        return actions, log_prob

    def sample_recurrent(self, obs, hidden):
        logits, hidden_out = self.architecture.architecture(obs, hidden)
        self.action_mean = logits.detach().cpu().numpy()
        actions, log_prob = self.distribution.sample(self.action_mean)
        return actions, log_prob, hidden_out

    def evaluate(self, obs, actions):
        if getattr(self.architecture, "is_recurrent", False):
            raise RuntimeError("Recurrent policy requires evaluate_recurrent()")
        self.action_mean = self.architecture.architecture(obs)
        return self.distribution.evaluate(self.action_mean, actions)

    def evaluate_recurrent(self, obs, actions, hidden):
        logits, hidden_out = self.architecture.architecture(obs, hidden)
        self.action_mean = logits
        logp, entropy = self.distribution.evaluate(self.action_mean, actions)
        return logp, entropy, hidden_out

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture.architecture(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    def update(self):
        self.distribution.update()

    def set_scripted_policy(self, scripted_policy):
        self.scripted_policy = scripted_policy

    def sync_scripted_policy(self):
        if self.scripted_policy is None:
            return
        self.scripted_policy.load_state_dict(self.architecture.architecture.state_dict())

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        if getattr(self.architecture, "is_recurrent", False):
            raise RuntimeError("Recurrent critic requires predict_recurrent()")
        return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        if getattr(self.architecture, "is_recurrent", False):
            raise RuntimeError("Recurrent critic requires evaluate_recurrent()")
        return self.architecture.architecture(obs)

    def predict_recurrent(self, obs, hidden):
        values, hidden_out = self.architecture.architecture(obs, hidden)
        return values.detach(), hidden_out

    def evaluate_recurrent(self, obs, hidden):
        values, hidden_out = self.architecture.architecture(obs, hidden)
        return values, hidden_out

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class DepthEncoderNet(nn.Module):
    def __init__(self, mlp_shape, activation_fn, proprio_dim, depth_height, depth_width,
                 depth_channels, depth_kernels, depth_strides, depth_latent_dim, output_size):
        super(DepthEncoderNet, self).__init__()
        if not (len(depth_channels) == len(depth_kernels) == len(depth_strides)):
            raise ValueError("Depth encoder channels/kernels/strides must have the same length.")

        self.activation_fn = activation_fn
        self.proprio_dim = proprio_dim
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.depth_size = depth_height * depth_width

        conv_modules = []
        in_ch = 1
        for out_ch, kernel, stride in zip(depth_channels, depth_kernels, depth_strides):
            conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride)
            conv_modules.append(conv)
            conv_modules.append(self.activation_fn())
            in_ch = out_ch
        self.depth_conv = nn.Sequential(*conv_modules)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, depth_height, depth_width)
            conv_out = self.depth_conv(dummy).view(1, -1)
            conv_out_dim = conv_out.shape[1]

        self.depth_fc = nn.Linear(conv_out_dim, depth_latent_dim)

        mlp_modules = [nn.Linear(depth_latent_dim + proprio_dim, mlp_shape[0]), self.activation_fn()]
        for idx in range(len(mlp_shape) - 1):
            mlp_modules.append(nn.Linear(mlp_shape[idx], mlp_shape[idx + 1]))
            mlp_modules.append(self.activation_fn())
        mlp_modules.append(nn.Linear(mlp_shape[-1], output_size))
        self.mlp = nn.Sequential(*mlp_modules)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, obs):
        original_shape = obs.shape
        if obs.dim() > 2:
            obs = obs.view(-1, obs.shape[-1])

        proprio = obs[:, :self.proprio_dim]
        depth = obs[:, self.proprio_dim:self.proprio_dim + self.depth_size]
        depth = depth.view(-1, 1, self.depth_height, self.depth_width)
        x = self.depth_conv(depth)
        x = x.view(x.size(0), -1)
        x = self.depth_fc(x)
        x = torch.cat((proprio, x), dim=1)
        out = self.mlp(x)
        if len(original_shape) > 2:
            return out.view(*original_shape[:-1], out.shape[-1])
        return out


class DepthEncoderMLP(nn.Module):
    def __init__(self, shape, activation_fn, proprio_dim, depth_height, depth_width,
                 depth_channels, depth_kernels, depth_strides, depth_latent_dim, output_size):
        super(DepthEncoderMLP, self).__init__()
        self.architecture = DepthEncoderNet(shape, activation_fn, proprio_dim, depth_height, depth_width,
                                            depth_channels, depth_kernels, depth_strides, depth_latent_dim, output_size)
        self.input_shape = [proprio_dim + depth_height * depth_width]
        self.output_shape = [output_size]


class DepthEncoderGRUNet(nn.Module):
    def __init__(self, mlp_shape, activation_fn, proprio_dim, depth_height, depth_width,
                 depth_channels, depth_kernels, depth_strides, depth_latent_dim, output_size, hidden_size):
        super(DepthEncoderGRUNet, self).__init__()
        if not (len(depth_channels) == len(depth_kernels) == len(depth_strides)):
            raise ValueError("Depth encoder channels/kernels/strides must have the same length.")

        self.activation_fn = activation_fn
        self.proprio_dim = proprio_dim
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.depth_size = depth_height * depth_width
        self.hidden_size = hidden_size

        conv_modules = []
        in_ch = 1
        for out_ch, kernel, stride in zip(depth_channels, depth_kernels, depth_strides):
            conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride)
            conv_modules.append(conv)
            conv_modules.append(self.activation_fn())
            in_ch = out_ch
        self.depth_conv = nn.Sequential(*conv_modules)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, depth_height, depth_width)
            conv_out = self.depth_conv(dummy).view(1, -1)
            conv_out_dim = conv_out.shape[1]

        self.depth_fc = nn.Linear(conv_out_dim, depth_latent_dim)
        self.gru = nn.GRU(depth_latent_dim + proprio_dim, hidden_size, batch_first=False)

        mlp_modules = []
        if len(mlp_shape) > 0:
            mlp_modules.append(nn.Linear(hidden_size, mlp_shape[0]))
            mlp_modules.append(self.activation_fn())
            for idx in range(len(mlp_shape) - 1):
                mlp_modules.append(nn.Linear(mlp_shape[idx], mlp_shape[idx + 1]))
                mlp_modules.append(self.activation_fn())
            mlp_modules.append(nn.Linear(mlp_shape[-1], output_size))
        else:
            mlp_modules.append(nn.Linear(hidden_size, output_size))
        self.head = nn.Sequential(*mlp_modules)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, obs, hidden=None):
        original_shape = obs.shape
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)  # [1, N, O]
        elif obs.dim() != 3:
            raise ValueError("GRU expects obs shape [N,O] or [T,N,O]")

        T, N, O = obs.shape
        obs_flat = obs.reshape(T * N, O)
        proprio = obs_flat[:, :self.proprio_dim]
        depth = obs_flat[:, self.proprio_dim:self.proprio_dim + self.depth_size]
        depth = depth.view(T * N, 1, self.depth_height, self.depth_width)
        x = self.depth_conv(depth)
        x = x.view(x.size(0), -1)
        x = self.depth_fc(x)
        x = torch.cat((proprio, x), dim=1)
        x = x.view(T, N, -1)

        if hidden is None:
            hidden = torch.zeros(1, N, self.hidden_size, device=obs.device)
        out, hidden_out = self.gru(x, hidden)
        out = self.head(out)

        if len(original_shape) == 2:
            return out.squeeze(0), hidden_out
        return out, hidden_out


class DepthEncoderGRU(nn.Module):
    def __init__(self, shape, activation_fn, proprio_dim, depth_height, depth_width,
                 depth_channels, depth_kernels, depth_strides, depth_latent_dim, output_size, hidden_size):
        super(DepthEncoderGRU, self).__init__()
        self.architecture = DepthEncoderGRUNet(shape, activation_fn, proprio_dim, depth_height, depth_width,
                                               depth_channels, depth_kernels, depth_strides, depth_latent_dim,
                                               output_size, hidden_size)
        self.input_shape = [proprio_dim + depth_height * depth_width]
        self.output_shape = [output_size]
        self.is_recurrent = True
        self.hidden_size = hidden_size


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, size, init_std, fast_sampler, seed=0):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        self.fast_sampler = fast_sampler
        self.fast_sampler.seed(seed)
        self.samples = np.zeros([size, dim], dtype=np.float32)
        self.logprob = np.zeros(size, dtype=np.float32)
        self.std_np = self.std.detach().cpu().numpy()

    def update(self):
        self.std_np = self.std.detach().cpu().numpy()

    def sample(self, logits):
        self.fast_sampler.sample(logits, self.std_np, self.samples, self.logprob)
        # Return views; RolloutStorage copies immediately.
        return self.samples, self.logprob

    def evaluate(self, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)
        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std
