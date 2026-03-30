## raisimGymTorch

Python + C++ (nanobind) bindings for RaiSim gym environments, plus training scripts.

### Dependencies
C++/build:
- CMake >= 3.10
- A C++17 compiler
- OpenMP
- Eigen3 (already vendored at `thirdParty/Eigen3`)
- raisim (provided in this repo under `raisim/`)

Python (runtime + training):
- Python >= 3.9 (nanobind requirement)
- See `requirements.txt`

### Virtual Environment (recommended)
From `raisimGymTorch/`:
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Python deps:
```
python -m pip install -r requirements.txt
```

Optional (stable-baselines3 example):
- gym
- stable-baselines3

### Build (uses the active Python)
From `raisimGymTorch/`:
```
python setup.py develop
```

Debug build:
```
python setup.py develop --Debug
```

If CMake cannot locate raisim or Eigen in a non-standard setup, pass:
```
python setup.py develop --CMAKE_PREFIX_PATH /path/to/prefix
```

### setup.py arguments
This project supports standard setuptools/distutils commands (e.g. `develop`, `build_ext`, `install`), plus the custom flags below. Custom flags must come after the command.

Custom flags:
- `--Debug`: build with `CMAKE_BUILD_TYPE=Debug` and create `<ENV_NAME>_debug_app`.
- `--CMAKE_PREFIX_PATH <path>`: forwarded to CMake for non-standard installs (e.g., raisim or Eigen).
- `--Libtorch`: enables libtorch support (`-DRAISIMGYM_TORCH_WITH_LIBTORCH=ON`).

If `--Libtorch` is set and no Torch path is provided, the build will try to download LibTorch into `thirdParty/libtorch`.
Control this with:
- `LIBTORCH_URL`: explicit LibTorch zip URL (overrides all other settings).
- `LIBTORCH_CHANNEL`: `nightly` (default) or `stable`.
- `LIBTORCH_CUDA`: `cpu` (default) or a CUDA tag like `cu118`.
- `LIBTORCH_VERSION`: required for `stable` channel (e.g., `2.8.0`).
- `LIBTORCH_SKIP_DOWNLOAD=1`: disable auto-download.

Examples:
```
python setup.py build_ext --inplace --Debug
python setup.py develop --CMAKE_PREFIX_PATH /path/to/prefix
python setup.py develop --Libtorch
```

### Run training (anymal example)
```
cd raisimGymTorch/env/envs/rsg_anymal
python runner.py
```

### Test a policy
```
python raisimGymTorch/env/envs/rsg_anymal/tester.py --weight data/roughTerrain/FOLDER_NAME/full_XXX.pt
```

### Retrain from a checkpoint
```
python raisimGymTorch/env/envs/rsg_anymal/runner.py --mode retrain --weight data/roughTerrain/FOLDER_NAME/full_XXX.pt
```

### Recurrent PPO (GRU) training flow
RNN training follows the split-and-pad strategy used in rsl_rl: trajectories are split at `done`, padded to a common
length, and masked so padded steps do not contribute to loss.

Data flow (rollout → storage):
```
for t in 0..T-1:
  obs_t        -> actor GRU -> action_t
  critic_obs_t -> critic GRU -> value_t
  store: obs_t, critic_obs_t, action_t, log_prob_t, reward_t, done_t
  store: actor_hidden_t, critic_hidden_t
```

Split and pad trajectories (done boundaries):
```
T x N rollout            ->         padded trajectories (max_len x num_traj)

env0: a1 a2 a3 a4 | a5 a6              a1 a2 a3 a4
env1: b1 b2 | b3 b4 b5 | b6      =>    a5 a6  0  0
                                        b1 b2  0  0
                                        b3 b4 b5 0
                                        b6  0  0  0

masks (same shape):
  T = valid, F = padded
  T T T T
  T T F F
  T T F F
  T T T F
  T F F F
```

Recurrent PPO update (masked):
```
for each mini-batch of trajectories:
  logits_seq, _ = actor GRU(padded_obs, init_hidden)
  values_seq, _ = critic GRU(padded_critic_obs, init_hidden)

  flatten -> apply mask -> compute PPO loss only on valid steps
```

This avoids per-timestep Python loops and keeps the GRU training efficient while handling episode boundaries correctly.

### Debugging
1. Build with debug symbols: `python setup.py develop --Debug` (this produces `<ENV_NAME>_debug_app`)
2. Run under Valgrind or a debugger (CLion works well).
