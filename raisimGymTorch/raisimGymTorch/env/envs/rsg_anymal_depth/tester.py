from ruamel.yaml import YAML
from raisimGymTorch.env.bin import rsg_anymal_depth
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import io
import math
import time
import torch
import argparse


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

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
cfg['environment']['num_envs'] = 1

env = VecEnv(rsg_anymal_depth.RaisimGymEnv(home_path + "/rsc", cfg_env_str), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
proprio_dim = cfg['environment']['proprio_dim']
depth_cfg = cfg['environment']['depth']
depth_width = depth_cfg['width']
depth_height = depth_cfg['height']
depth_size = depth_width * depth_height
expected_ob_dim = proprio_dim + depth_size
if ob_dim != expected_ob_dim:
    raise RuntimeError(f"Observation dim mismatch: env reports {ob_dim}, "
                       f"but proprio+depth is {expected_ob_dim}.")

depth_arch = cfg['architecture']['depth_encoder']
gru_hidden = cfg['architecture'].get('gru_hidden', 128)

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.DepthEncoderGRU(cfg['architecture']['policy_net'],
                                              torch.nn.LeakyReLU,
                                              proprio_dim,
                                              depth_height,
                                              depth_width,
                                              depth_arch['channels'],
                                              depth_arch['kernels'],
                                              depth_arch['strides'],
                                              depth_arch['latent_dim'],
                                              act_dim,
                                              gru_hidden)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    # max_steps = 1000000
    max_steps = 1000 ## 10 secs

    hidden = torch.zeros(1, 1, gru_hidden, dtype=torch.float32)
    for step in range(max_steps):
        time.sleep(0.01)
        obs = env.observe(False)
        obs_tensor = torch.from_numpy(obs).cpu()
        action_ll, hidden = loaded_graph.architecture(obs_tensor, hidden)
        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
        reward_ll_sum = reward_ll_sum + reward_ll[0]
        if dones:
            hidden.zero_()
        if dones or step == max_steps - 1:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
            print('----------------------------------------------------\n')
            start_step_id = step + 1
            reward_ll_sum = 0.0

    env.turn_off_visualization()
    env.reset()
    print("Finished at the maximum visualization steps")
