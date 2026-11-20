## raisimGymTorch

Python and C++ bindings for training RaiSim environments with PyTorch.

### Requirements

- Python 3.9 or newer, with development headers (e.g. `python3.X-dev` on Ubuntu)
- CMake 3.10 or newer
- A C++20 compiler
- OpenMP (included with GCC; with Clang, install `libomp-dev`)

RaiSim, rayrai, Eigen, and nanobind are included in the parent repository.

### Install and build

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

This creates an optimized Release build and installs the package in editable
mode. `--no-build-isolation` makes the build use the PyTorch installation from
the active environment. Additional compiler flags are not required. After
changing C++ code, run the same `pip install` command again to rebuild.

### Train ANYmal

```bash
python raisimGymTorch/env/envs/rsg_anymal/runner.py --mode train
```

The runner automatically uses the optimized C++ environment, C++ return
calculation, CPU thread settings, and C++ policy evaluation.
If the C++ TorchScript evaluator is unavailable, evaluation automatically falls
back to PyTorch on CUDA when available, otherwise CPU.

To use another network, change `build_actor_network()` and
`build_critic_network()` in `runner.py`. If it needs a special evaluation
wrapper, also change `build_evaluation_network()`. No C++ changes are needed.

Resume from a checkpoint:

```bash
python raisimGymTorch/env/envs/rsg_anymal/runner.py \
  --mode retrain --weight /path/to/full_N.pt
```

Test a checkpoint:

```bash
python raisimGymTorch/env/envs/rsg_anymal/tester.py \
  --weight /path/to/full_N.pt
```

### Train ANYmal on rough terrain

```bash
python raisimGymTorch/env/envs/rsg_anymal_terrain/runner.py --mode train
```

All environments share one heightmap, which is replaced by a new random one
every 10 iterations (`terrain/randomize_every_n` in its `cfg.yaml`). Its
`tester.py` and `--mode retrain` work as above.

### Benchmark

```bash
python raisimGymTorch/env/envs/rsg_anymal/benchmark_rollout.py
```
