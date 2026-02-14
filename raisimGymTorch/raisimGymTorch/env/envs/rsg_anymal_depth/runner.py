from ruamel.yaml import YAML
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.rsg_anymal_depth import NormalSampler
from raisimGymTorch.env.bin.rsg_anymal_depth import RaisimGymEnv
from raisimGymTorch.env.RewardAnalyzer import RewardAnalyzer
import os
import io
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse


# task specification
task_name = "anymal_depth_locomotion"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('--export_onnx', action='store_true', help='export ONNX models at startup')
parser.add_argument('--no_tb', action='store_true', help='disable tensorboard launch')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[RAISIM_GYM] Using device: {device}")

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
critic_ob_dim = env.num_critic_obs
act_dim = env.num_acts
num_threads = cfg['environment']['num_threads']
proprio_dim = cfg['environment']['proprio_dim']
depth_cfg = cfg['environment']['depth']
depth_width = depth_cfg['width']
depth_height = depth_cfg['height']
depth_size = depth_width * depth_height
expected_ob_dim = proprio_dim + depth_size
if ob_dim != expected_ob_dim:
    raise RuntimeError(f"Observation dim mismatch: env reports {ob_dim}, "
                       f"but proprio+depth is {expected_ob_dim}.")
expected_critic_dim = proprio_dim + 50 * 50
if critic_ob_dim != expected_critic_dim:
    raise RuntimeError(f"Critic obs dim mismatch: env reports {critic_ob_dim}, "
                       f"but proprio+heightpatch is {expected_critic_dim}.")

depth_arch = cfg['architecture']['depth_encoder']
gru_hidden = cfg['architecture'].get('gru_hidden', 128)

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []
curriculum_level = 0.0

actor_arch = ppo_module.DepthEncoderGRU(cfg['architecture']['policy_net'],
                                        nn.LeakyReLU,
                                        proprio_dim,
                                        depth_height,
                                        depth_width,
                                        depth_arch['channels'],
                                        depth_arch['kernels'],
                                        depth_arch['strides'],
                                        depth_arch['latent_dim'],
                                        act_dim,
                                        gru_hidden)
critic_arch = ppo_module.DepthEncoderGRU(cfg['architecture']['value_net'],
                                         nn.LeakyReLU,
                                         proprio_dim,
                                         50,
                                         50,
                                         depth_arch['channels'],
                                         depth_arch['kernels'],
                                         depth_arch['strides'],
                                         depth_arch['latent_dim'],
                                         1,
                                         gru_hidden)

actor = ppo_module.Actor(actor_arch,
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           1.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(critic_arch, device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])

if args.export_onnx or os.environ.get("RAISIMGYM_EXPORT_ONNX", "0") == "1":
    # Export policy/critic to ONNX for Netron visualization (before training).
    actor_arch.architecture.eval()
    critic_arch.architecture.eval()
    actor_onnx_path = os.path.join(saver.data_dir, "actor.onnx")
    critic_onnx_path = os.path.join(saver.data_dir, "critic.onnx")
    with torch.no_grad():
        actor_dummy_obs = torch.zeros(1, ob_dim, dtype=torch.float32)
        actor_dummy_h0 = torch.zeros(1, 1, gru_hidden, dtype=torch.float32)
        torch.onnx.export(
            actor_arch.architecture.cpu(),
            (actor_dummy_obs, actor_dummy_h0),
            actor_onnx_path,
            opset_version=18,
            dynamo=False,
            do_constant_folding=True,
            input_names=["obs", "h0"],
            output_names=["action", "h1"],
            dynamic_axes={"obs": {0: "batch"}, "h0": {1: "batch"},
                          "action": {0: "batch"}, "h1": {1: "batch"}},
        )

        critic_dummy_obs = torch.zeros(1, critic_ob_dim, dtype=torch.float32)
        critic_dummy_h0 = torch.zeros(1, 1, gru_hidden, dtype=torch.float32)
        torch.onnx.export(
            critic_arch.architecture.cpu(),
            (critic_dummy_obs, critic_dummy_h0),
            critic_onnx_path,
            opset_version=18,
            dynamo=False,
            do_constant_folding=True,
            input_names=["obs", "h0"],
            output_names=["value", "h1"],
            dynamic_axes={"obs": {0: "batch"}, "h0": {1: "batch"},
                          "value": {0: "batch"}, "h1": {1: "batch"}},
        )
    print(f"[RAISIM_GYM] Saved ONNX policy to {actor_onnx_path}")
    print(f"[RAISIM_GYM] Saved ONNX critic to {critic_onnx_path}")
    # Move back to training device (export moves modules to CPU).
    actor.architecture.to(device)
    critic.architecture.to(device)
    actor.architecture.train()
    critic.architecture.train()

if (not args.no_tb) and os.environ.get("RAISIMGYM_SKIP_TB", "0") != "1":
    tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=cfg.get('algorithm', {}).get('num_learning_epochs', 4),
              gamma=0.996,
              lam=0.95,
              num_mini_batches=cfg.get('algorithm', {}).get('num_mini_batches', 4),
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              )

reward_analyzer = RewardAnalyzer(env, ppo.writer)

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

for update in range(1000000):
    start = time.time()
    env.reset()
    reward_sum = 0
    done_sum = 0
    average_dones = 0.

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.DepthEncoderGRU(cfg['architecture']['policy_net'],
                                                  nn.LeakyReLU,
                                                  proprio_dim,
                                                  depth_height,
                                                  depth_width,
                                                  depth_arch['channels'],
                                                  depth_arch['kernels'],
                                                  depth_arch['strides'],
                                                  depth_arch['latent_dim'],
                                                  act_dim,
                                                  gru_hidden)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        actor.architecture.eval()
        critic.architecture.eval()
        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        hidden = torch.zeros(1, env.num_envs, gru_hidden, dtype=torch.float32)
        for step in range(n_steps):
            with torch.no_grad():
                frame_start = time.time()
                obs = env.observe(False)
                critic_obs = env.observe_critic()
                obs_tensor = torch.from_numpy(obs).cpu()
                action, hidden = loaded_graph.architecture(obs_tensor, hidden)
                reward, dones = env.step(action.cpu().detach().numpy())
                if np.any(dones):
                    done_idx = np.where(dones)[0]
                    hidden[:, done_idx, :] = 0.0
                reward_analyzer.add_reward_info(env.get_reward_info())
                frame_end = time.time()
                wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                if wait_time > 0.:
                    time.sleep(wait_time)

        env.stop_video_recording()
        env.turn_off_visualization()
        actor.architecture.train()
        critic.architecture.train()

        reward_analyzer.analyze_and_plot(update)
        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    # actual training
    for step in range(n_steps):
        obs = env.observe()
        critic_obs = env.observe_critic()
        action = ppo.act(obs)
        reward, dones = env.step(action)
        ppo.step(value_obs=critic_obs, rews=reward, dones=dones)
        done_sum = done_sum + np.sum(dones)
        reward_sum = reward_sum + np.sum(reward)

    # take st step to get value obs
    obs = env.observe()
    critic_obs = env.observe_critic()
    ppo.update(actor_obs=obs, value_obs=critic_obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.update()
    actor.distribution.enforce_minimum_std((torch.ones(act_dim) * 0.2).to(device))

    # curriculum update
    if update % 50 == 0 and average_dones < 0.001:
        env.curriculum_callback()
        curriculum_level += 0.1

    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("curriculum z-scale: ", '{:0.2f}'.format(curriculum_level)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('----------------------------------------------------\n')
