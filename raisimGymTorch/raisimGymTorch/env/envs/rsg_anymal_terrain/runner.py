import argparse
import copy
import io
import math
import os
import sys
import time

# libgomp reads these settings when its first worker pool is created. Binding
# update workers to physical cores improves the policy/value backward passes and
# setdefault preserves an explicit launch-time choice by the user.
if sys.platform.startswith('linux'):
    os.environ.setdefault('OMP_PROC_BIND', 'close')
    os.environ.setdefault('OMP_PLACES', 'cores')
os.environ.setdefault('OMP_DYNAMIC', 'FALSE')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import torch
import torch.nn as nn
from ruamel.yaml import YAML
from torch.distributions import Normal

import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.env.RewardAnalyzer import RewardAnalyzer
from raisimGymTorch.env.bin import rsg_anymal_terrain
from raisimGymTorch.helper.raisim_gym_helper import (
    ConfigurationSaver, load_param, save_scripted_policy, script_policy, tensorboard_launcher)


# task specification
task_name = "anymal_terrain_locomotion"


def activation_fn():
    return nn.LeakyReLU(inplace=True)


def build_actor_network(observation_dim, action_dim):
    """Change this function to use another Python-defined policy network."""
    return ppo_module.MLP(
        cfg['architecture']['policy_net'], activation_fn,
        observation_dim, action_dim)


def build_critic_network(observation_dim):
    """Change this function to use another Python-defined value network."""
    return ppo_module.MLP(
        cfg['architecture']['value_net'], activation_fn,
        observation_dim, 1)


def build_evaluation_network(actor_network, evaluation_device):
    """Return a module mapping a batch of observations to actions."""
    return copy.deepcopy(actor_network.architecture).to(evaluation_device).eval()


class ZeroEntropyGaussian(ppo_module.MultivariateGaussianDiagonalCovariance):
    """Avoid materializing entropy when PPO's entropy coefficient is zero."""

    def evaluate(self, logits, outputs):
        # std is constrained to remain positive. Disabling validation skips
        # repeated support/constraint scans while retaining identical math.
        distribution = Normal(logits, self.std.reshape(self.dim), validate_args=False)
        return distribution.log_prob(outputs).sum(dim=1), logits.new_zeros(())

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('--eval-backend', choices=('auto', 'cpp', 'python'), default='auto',
                    help='run deterministic visualization rollouts in one C++ call when available')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

NormalSampler = rsg_anymal_terrain.NormalSampler
RaisimGymEnv = rsg_anymal_terrain.RaisimGymEnv
CppTorchPolicyRunner = getattr(rsg_anymal_terrain, 'TorchPolicyRunner', None)

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[RAISIM_GYM] Using device: {device}")

# The policy is small, while RaiSim already occupies most CPU cores. Letting
# PyTorch use its default full CPU pool causes severe nested-thread contention
# during rollout. PPO updates run while RaiSim is idle and benefit from a larger
# pool, so use phase-specific thread counts.
if device.type == 'cpu':
    rollout_torch_threads = 1
    update_torch_threads = min(12, os.cpu_count() or 1)
    torch.set_num_threads(rollout_torch_threads)

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
yaml = YAML()
cfg = yaml.load(open(task_path + "/cfg.yaml", 'r'))
cfg_stream = io.StringIO()
yaml.dump(cfg['environment'], cfg_stream)
cfg_env_str = cfg_stream.getvalue()

# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/rsc", cfg_env_str))
env.seed(cfg['seed'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

actor = ppo_module.Actor(build_actor_network(ob_dim, act_dim),
                         ZeroEntropyGaussian(act_dim,
                                             env.num_envs,
                                             1.0,
                                             NormalSampler(act_dim),
                                             cfg['seed']),
                         device)
critic = ppo_module.Critic(build_critic_network(ob_dim), device)
minimum_action_std = torch.full((act_dim,), 0.2, device=device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              entropy_coef=0.0,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              shared_observations=True,
              returns_calculator=getattr(rsg_anymal_terrain, 'compute_returns', None),
              # Compiling the update pays off only once a minibatch is large
              # enough to amortize the compiled dispatch: measured 9% faster at
              # 40k samples per minibatch and 20% slower at 10k.
              compile_networks=False,
              )

reward_analyzer = RewardAnalyzer(env, ppo.writer)
reward_info_history = np.empty(
    (n_steps, env.num_envs, len(env.get_reward_names())), dtype=np.float32)
if args.eval_backend != 'python':
    if CppTorchPolicyRunner is not None:
        print('[RAISIM_GYM] Using single-call C++ evaluation with the Python TorchScript policy')
    else:
        print('[RAISIM_GYM] C++ TorchScript unavailable; using Python evaluation')

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

if device.type == 'cpu':
    # TorchScript removes Python module dispatch from the 400 inference-only
    # policy calls. PPO evaluation/backpropagation still uses the original
    # eager module and parameters.
    try:
        actor.set_scripted_policy(script_policy(actor.architecture.architecture))
    except Exception as error:
        print(f'[RAISIM_GYM] Policy is not TorchScript compatible; using eager rollout: {error}')

for update in range(1000000):
    start = time.time()
    env.reset()
    reward_sum = 0
    done_sum = 0

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        evaluation_policy = build_evaluation_network(actor.architecture, device)
        scripted_evaluation_policy = None
        if CppTorchPolicyRunner is not None and args.eval_backend != 'python':
            try:
                cpp_evaluation_policy = build_evaluation_network(
                    actor.architecture, 'cpu')
                scripted_evaluation_policy = script_policy(cpp_evaluation_policy)
            except Exception as error:
                print(f'[RAISIM_GYM] C++ policy export failed; using Python evaluation: {error}')

        env.turn_on_visualization()
        # rayrai renders the video in this process and saves it next to the checkpoint.
        video_path = os.path.abspath(os.path.join(saver.data_dir, "policy_"+str(update)+'.mp4'))
        recording = cfg['record_video'] and env.start_video_recording(video_path)
        if cfg['record_video'] and not recording:
            print('[RAISIM_GYM] Not recording this evaluation: ' + env.get_video_status())
        # The video is timed by simulation time, so run in real time only for a connected viewer.
        playback_dt = cfg['environment']['control_dt'] if env.is_visualizer_connected() else 0.

        evaluated_in_cpp = False
        if (CppTorchPolicyRunner is not None and
                scripted_evaluation_policy is not None and
                args.eval_backend != 'python'):
            # Serializing and loading the policy is the only part of training
            # that depends on TorchScript surviving. Falling back keeps a future
            # removal from ending the run at its first evaluation.
            try:
                policy_path = os.path.join(saver.data_dir, 'evaluation_policy.pt')
                save_scripted_policy(scripted_evaluation_policy, policy_path)
                cpp_policy_runner = CppTorchPolicyRunner(env.wrapper, policy_path)
                cpp_policy_runner.run(reward_info_history, playback_dt)
                reward_analyzer.add_reward_info(
                    reward_info_history.reshape(-1, reward_info_history.shape[-1]))
                evaluated_in_cpp = True
            except Exception as error:
                print(f'[RAISIM_GYM] C++ evaluation failed; evaluating in Python: {error}')

        if not evaluated_in_cpp:
            with torch.inference_mode():
                for step in range(n_steps):
                    frame_start = time.time()
                    obs = env.observe(False)
                    obs_tensor = torch.from_numpy(obs).to(device)
                    action = evaluation_policy(obs_tensor)
                    reward, dones = env.step(action.cpu().numpy())
                    reward_analyzer.add_reward_info(env.get_reward_info())
                    frame_end = time.time()
                    wait_time = playback_dt - (frame_end-frame_start)
                    if wait_time > 0.:
                        time.sleep(wait_time)

        if recording:
            if env.stop_video_recording():
                print('[RAISIM_GYM] Saved the evaluation video: ' + video_path)
            else:
                print('[RAISIM_GYM] The evaluation video failed: ' + env.get_video_status())
        env.turn_off_visualization()

        reward_analyzer.analyze_and_plot(update)
        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    # actual training
    obs = env.observe()
    with torch.inference_mode():
        for step in range(n_steps):
            action = ppo.act_inference(obs)
            reward, dones, next_obs = env.step_and_observe(action)
            ppo.step(value_obs=obs, rews=reward, dones=dones)
            done_sum += np.count_nonzero(dones)
            reward_sum += np.sum(reward)
            obs = next_obs

    # obs already contains the post-rollout value observation.
    if device.type == 'cpu':
        torch.set_num_threads(update_torch_threads)
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_sum / total_steps
    average_dones = done_sum / total_steps

    actor.update()
    actor.distribution.enforce_minimum_std(minimum_action_std)
    actor.sync_scripted_policy()
    if device.type == 'cpu':
        torch.set_num_threads(rollout_torch_threads)
        # torch.set_num_threads() also changes the process-wide OpenMP setting
        # used by RaiSim. Restore the independently tuned simulator pool.
        env.set_num_threads(cfg['environment']['num_threads'])

    # curriculum update. Every terrain/randomize_every_n calls (iterations),
    # Environment.hpp moves all environments onto a new shared heightmap.
    env.curriculum_callback()

    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('----------------------------------------------------\n')
