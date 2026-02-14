//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#ifdef RAISIMGYM_TORCH_WITH_LIBTORCH
#include <torch/script.h>
#include <torch/torch.h>
#include <unordered_map>
#include <cstring>
#endif
#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace py = nanobind;
namespace nb = nanobind;
using namespace raisim;
int THREAD_COUNT = 1;

#ifndef ENVIRONMENT_NAME
  #define ENVIRONMENT_NAME RaisimGymEnv
#endif

NB_MODULE(RAISIMGYM_TORCH_ENV_NAME, m) {
  using EnvType = VectorizedEnvironment<ENVIRONMENT, ENVIRONMENT::kStaticSchedule>;
  py::class_<EnvType>(m, RSG_MAKE_STR(ENVIRONMENT_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &EnvType::init)
    .def("reset", &EnvType::reset)
    .def("observe", &EnvType::observe)
    .def("observeCritic", &EnvType::observeCritic)
    .def("step", &EnvType::step)
    .def("setSeed", &EnvType::setSeed)
    .def("getRewardInfo", &EnvType::getRewardInfo, py::rv_policy::reference_internal)
    .def("getRewardNames", &EnvType::getRewardNames)
    .def("close", &EnvType::close)
    .def("isTerminalState", &EnvType::isTerminalState)
    .def("setSimulationTimeStep", &EnvType::setSimulationTimeStep)
    .def("setControlTimeStep", &EnvType::setControlTimeStep)
    .def("getObDim", &EnvType::getObDim)
    .def("getCriticObDim", &EnvType::getCriticObDim)
    .def("getActionDim", &EnvType::getActionDim)
    .def("getNumOfEnvs", &EnvType::getNumOfEnvs)
    .def("turnOnVisualization", &EnvType::turnOnVisualization)
    .def("turnOffVisualization", &EnvType::turnOffVisualization)
    .def("stopRecordingVideo", &EnvType::stopRecordingVideo)
    .def("startRecordingVideo", &EnvType::startRecordingVideo)
    .def("curriculumUpdate", &EnvType::curriculumUpdate)
    .def("getObStatistics", &EnvType::getObStatistics)
    .def("setObStatistics", &EnvType::setObStatistics)
    .def("__getstate__", [](const EnvType &p) {
        return py::make_tuple(p.getResourceDir(), p.getCfgString());
    })
    .def("__setstate__", [](EnvType &p, py::tuple t) {
        if (t.size() != 2) {
            throw std::runtime_error("Invalid state!");
        }

        new (&p) EnvType(py::cast<std::string>(t[0]),
                                                    py::cast<std::string>(t[1]));
    });

  py::class_<NormalSampler>(m, "NormalSampler")
    .def(py::init<int>(), py::arg("dim"))
    .def("seed", &NormalSampler::seed)
    .def("sample", &NormalSampler::sample);

#ifdef RAISIMGYM_TORCH_WITH_LIBTORCH
  using nb::c_contig;
  using nb::device::cpu;

  class TorchRolloutRunner {
   public:
    TorchRolloutRunner(EnvType &env, const std::string &module_path, int seed = 0)
        : env_(env), sampler_(env.getActionDim()) {
      sampler_.seed(seed);
      module_ = torch::jit::load(module_path);
      module_.eval();
      module_.to(at::kCPU);
    }

    void update_weights(nb::dict weights) {
      torch::NoGradGuard no_grad;
      auto params = module_.named_parameters(/*recurse=*/true);
      std::unordered_map<std::string, torch::Tensor> param_map;
      param_map.reserve(params.size());
      for (const auto &p : params) {
        param_map.emplace(p.name, p.value);
      }

      for (auto item : weights) {
        const std::string name = nb::cast<std::string>(item.first);
        auto it = param_map.find(name);
        if (it == param_map.end()) {
          throw std::runtime_error("Unknown parameter name in state dict: " + name);
        }
        auto arr = nb::cast<nb::ndarray<float, c_contig, cpu>>(item.second);
        std::vector<int64_t> sizes;
        sizes.reserve(arr.ndim());
        for (size_t i = 0; i < arr.ndim(); ++i) {
          sizes.push_back(static_cast<int64_t>(arr.shape(i)));
        }
        auto src = torch::from_blob(arr.data(), sizes, torch::TensorOptions().dtype(torch::kFloat32));
        it->second.copy_(src);
      }
    }

    void rollout(nb::ndarray<float, c_contig, cpu> actor_obs,
                 nb::ndarray<float, c_contig, cpu> critic_obs,
                 nb::ndarray<float, c_contig, cpu> actions,
                 nb::ndarray<float, c_contig, cpu> rewards,
                 nb::ndarray<bool, c_contig, cpu> dones,
                 nb::ndarray<float, c_contig, cpu> mu,
                 nb::ndarray<float, c_contig, cpu> sigma,
                 nb::ndarray<float, c_contig, cpu> logp,
                 nb::ndarray<float, c_contig, cpu> std_vec,
                 bool update_obs_stats) {
      if (actor_obs.ndim() != 3) {
        throw std::runtime_error("actor_obs must be 3D [T, N, O]");
      }
      if (critic_obs.ndim() != 3) {
        throw std::runtime_error("critic_obs must be 3D [T, N, O]");
      }
      if (actions.ndim() != 3) {
        throw std::runtime_error("actions must be 3D [T, N, A]");
      }
      if (mu.ndim() != 3 || sigma.ndim() != 3) {
        throw std::runtime_error("mu/sigma must be 3D [T, N, A]");
      }
      if (rewards.ndim() != 3 || dones.ndim() != 3 || logp.ndim() != 3) {
        throw std::runtime_error("rewards/dones/logp must be 3D [T, N, 1]");
      }
      if (std_vec.ndim() != 1) {
        throw std::runtime_error("std_vec must be 1D [A]");
      }

      const int64_t T = static_cast<int64_t>(actor_obs.shape(0));
      const int64_t N = static_cast<int64_t>(actor_obs.shape(1));
      const int64_t O = static_cast<int64_t>(actor_obs.shape(2));
      const int64_t A = static_cast<int64_t>(actions.shape(2));

      if (critic_obs.shape(0) != actor_obs.shape(0) ||
          critic_obs.shape(1) != actor_obs.shape(1) ||
          critic_obs.shape(2) != actor_obs.shape(2)) {
        throw std::runtime_error("actor_obs and critic_obs shapes must match");
      }
      if (actions.shape(0) != actor_obs.shape(0) ||
          actions.shape(1) != actor_obs.shape(1)) {
        throw std::runtime_error("actions shape mismatch");
      }
      if (mu.shape(0) != actor_obs.shape(0) || mu.shape(1) != actor_obs.shape(1) || mu.shape(2) != actions.shape(2) ||
          sigma.shape(0) != actor_obs.shape(0) || sigma.shape(1) != actor_obs.shape(1) || sigma.shape(2) != actions.shape(2)) {
        throw std::runtime_error("mu/sigma shape mismatch");
      }
      if (rewards.shape(0) != actor_obs.shape(0) || rewards.shape(1) != actor_obs.shape(1) || rewards.shape(2) != 1) {
        throw std::runtime_error("rewards shape mismatch");
      }
      if (dones.shape(0) != actor_obs.shape(0) || dones.shape(1) != actor_obs.shape(1) || dones.shape(2) != 1) {
        throw std::runtime_error("dones shape mismatch");
      }
      if (logp.shape(0) != actor_obs.shape(0) || logp.shape(1) != actor_obs.shape(1) || logp.shape(2) != 1) {
        throw std::runtime_error("logp shape mismatch");
      }
      if (std_vec.shape(0) != static_cast<size_t>(A)) {
        throw std::runtime_error("std_vec length mismatch");
      }

      float *actor_ptr = actor_obs.data();
      float *critic_ptr = critic_obs.data();
      float *action_ptr = actions.data();
      float *reward_ptr = rewards.data();
      bool *done_ptr = dones.data();
      float *mu_ptr = mu.data();
      float *sigma_ptr = sigma.data();
      float *logp_ptr = logp.data();
      float *std_ptr = std_vec.data();

      Eigen::Map<EigenVec> std_map(std_ptr, A);

      nb::gil_scoped_release release;
      torch::NoGradGuard no_grad;

      for (int64_t t = 0; t < T; ++t) {
        Eigen::Map<EigenRowMajorMat> obs_map(actor_ptr + t * N * O, N, O);
        env_.observe(obs_map, update_obs_stats);

        if (critic_ptr != actor_ptr) {
          std::memcpy(critic_ptr + t * N * O, obs_map.data(), sizeof(float) * N * O);
        }

        auto obs_tensor = torch::from_blob(obs_map.data(), {N, O}, torch::TensorOptions().dtype(torch::kFloat32));
        auto out = module_.forward({obs_tensor}).toTensor().contiguous();
        float *mean_ptr = out.data_ptr<float>();

        std::memcpy(mu_ptr + t * N * A, mean_ptr, sizeof(float) * N * A);
        for (int64_t i = 0; i < N; ++i) {
          std::memcpy(sigma_ptr + t * N * A + i * A, std_ptr, sizeof(float) * A);
        }

        Eigen::Map<EigenRowMajorMat> mean_map(mean_ptr, N, A);
        Eigen::Map<EigenRowMajorMat> action_map(action_ptr + t * N * A, N, A);
        Eigen::Map<EigenVec> logp_map(logp_ptr + t * N, N);
        sampler_.sample(mean_map, std_map, action_map, logp_map);

        Eigen::Map<EigenVec> reward_map(reward_ptr + t * N, N);
        Eigen::Map<EigenBoolVec> done_map(done_ptr + t * N, N);
        env_.step(action_map, reward_map, done_map);
      }
    }

   private:
    EnvType &env_;
    NormalSampler sampler_;
    torch::jit::Module module_;
  };

  py::class_<TorchRolloutRunner>(m, "TorchRolloutRunner")
    .def(py::init<EnvType &, const std::string &, int>(),
         py::arg("env"), py::arg("module_path"), py::arg("seed") = 0)
    .def("update_weights", &TorchRolloutRunner::update_weights, py::arg("state_dict"))
    .def("rollout", &TorchRolloutRunner::rollout,
         py::arg("actor_obs"),
         py::arg("critic_obs"),
         py::arg("actions"),
         py::arg("rewards"),
         py::arg("dones"),
         py::arg("mu"),
         py::arg("sigma"),
         py::arg("logp"),
         py::arg("std_vec"),
         py::arg("update_obs_stats") = true);
#endif
}
