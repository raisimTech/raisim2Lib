//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include "RaisimGymEnv.hpp"
#ifdef RAISIMGYM_NO_OPENMP
#ifndef RAISIMGYM_OPENMP_FALLBACK_HPP
#define RAISIMGYM_OPENMP_FALLBACK_HPP
inline void omp_set_num_threads(int) {}
inline int omp_get_thread_num() { return 0; }
#endif
#else
#include "omp.h"
#endif
#include "Yaml.hpp"
#ifdef RAISIMGYM_WITH_RAYRAI
#include "VideoRecorder.hpp"
#endif

namespace raisim {

int THREAD_COUNT;

template<class ChildEnvironment, bool StaticSchedule = false, bool ParallelizeVisualEnvironment = false>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg, bool normalizeObservation=true)
      : resourceDir_(resourceDir), cfgString_(cfg), normalizeObservation_(normalizeObservation) {
    Yaml::Parse(cfg_, cfg);

    if(&cfg_["render"])
      render_ = cfg_["render"].template As<bool>();
    init();
  }

  ~VectorizedEnvironment() {
#ifdef RAISIMGYM_WITH_RAYRAI
    videoRecorder_.reset();  // its renderer reads environment 0's world
#endif
    for (auto *ptr: environments_)
      delete ptr;
  }

  const std::string& getResourceDir() const { return resourceDir_; }
  const std::string& getCfgString() const { return cfgString_; }

  void init() {
    setNumThreads(cfg_["num_threads"].template As<int>());
    num_envs_ = cfg_["num_envs"].template As<int>();

    environments_.reserve(num_envs_);
    for (int i = 0; i < num_envs_; i++) {
      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
      environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
      environments_.back()->setControlTimeStep(cfg_["control_dt"].template As<double>());
    }

    for (int i = 0; i < num_envs_; i++) {
      // only the first environment is visualized
      environments_[i]->init();
      environments_[i]->getWorld()->setSensorNoiseSeed(uint64_t(i) + 1);
      environments_[i]->reset();
    }

    obDim_ = environments_[0]->getObDim();
    criticObDim_ = environments_[0]->getCriticObDim();
    actionDim_ = environments_[0]->getActionDim();
    rewardDim_ = environments_[0]->getRewards().getSize();
    rewardNames_ = environments_[0]->getRewards().getNames();
    RSFATAL_IF(obDim_ == 0 || actionDim_ == 0, "Observation/Action dimension must be defined in the constructor of each environment!")
    RSFATAL_IF(rewardDim_ == 0, "Reward list must be defined in the constructor of each environment!")

    rewardInformation_.setZero(num_envs_, rewardDim_);

    /// ob scaling
    if (normalizeObservation_) {
      obMean_.setZero(obDim_);
      obVar_.setOnes(obDim_);
      recentMean_.setZero(obDim_);
      recentVar_.setZero(obDim_);
      delta_.setZero(obDim_);
      epsilon.setZero(obDim_);
      epsilon.setConstant(1e-8);
    }
  }

  // resets all environments and returns observation
  void reset() {
    for (auto env: environments_)
      env->reset();
  }

  void observe(Eigen::Ref<EigenRowMajorMat> ob, bool updateStatistics) {
    if (render_ && num_envs_ > 0 && !ParallelizeVisualEnvironment) {
      environments_[0]->observe(ob.row(0));
      if constexpr (StaticSchedule) {
#pragma omp parallel for schedule(static)
        for (int i = 1; i < num_envs_; i++)
          environments_[i]->observe(ob.row(i));
      } else {
#pragma omp parallel for schedule(auto)
        for (int i = 1; i < num_envs_; i++)
          environments_[i]->observe(ob.row(i));
      }
    } else if constexpr (StaticSchedule) {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < num_envs_; i++)
        environments_[i]->observe(ob.row(i));
    } else {
#pragma omp parallel for schedule(auto)
      for (int i = 0; i < num_envs_; i++)
        environments_[i]->observe(ob.row(i));
    }

    if (normalizeObservation_)
      updateObservationStatisticsAndNormalize(ob, updateStatistics);
  }

  void observeCritic(Eigen::Ref<EigenRowMajorMat> ob) {
    if (render_ && num_envs_ > 0) {
      environments_[0]->observeCritic(ob.row(0));
      if constexpr (StaticSchedule) {
#pragma omp parallel for schedule(static)
        for (int i = 1; i < num_envs_; i++)
          environments_[i]->observeCritic(ob.row(i));
      } else {
#pragma omp parallel for schedule(auto)
        for (int i = 1; i < num_envs_; i++)
          environments_[i]->observeCritic(ob.row(i));
      }
    } else {
      if constexpr (StaticSchedule) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < num_envs_; i++)
          environments_[i]->observeCritic(ob.row(i));
      } else {
#pragma omp parallel for schedule(auto)
        for (int i = 0; i < num_envs_; i++)
          environments_[i]->observeCritic(ob.row(i));
      }
    }
  }


  void step(Eigen::Ref<EigenRowMajorMat> action,
            Eigen::Ref<EigenVec> reward,
            Eigen::Ref<EigenBoolVec> done) {
    if (render_ && num_envs_ > 0 && !ParallelizeVisualEnvironment) {
      perAgentStep(0, action, reward, done);
      if constexpr (StaticSchedule) {
#pragma omp parallel for schedule(static)
        for (int i = 1; i < num_envs_; i++)
          perAgentStep(i, action, reward, done);
      } else {
#pragma omp parallel for schedule(auto)
        for (int i = 1; i < num_envs_; i++)
          perAgentStep(i, action, reward, done);
      }
    } else if constexpr (StaticSchedule) {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < num_envs_; i++)
        perAgentStep(i, action, reward, done);
    } else {
#pragma omp parallel for schedule(auto)
      for (int i = 0; i < num_envs_; i++)
        perAgentStep(i, action, reward, done);
    }
    captureVideoFrame();
  }

  /// Advance every environment and produce the following normalized
  /// observation in the same OpenMP region. This is equivalent to step()
  /// followed by observe(), but avoids launching a second worker team merely
  /// to copy each environment's cached observation.
  void stepAndObserve(Eigen::Ref<EigenRowMajorMat> action,
                      Eigen::Ref<EigenVec> reward,
                      Eigen::Ref<EigenBoolVec> done,
                      Eigen::Ref<EigenRowMajorMat> observation,
                      bool updateStatistics) {
    if (render_ && num_envs_ > 0 && !ParallelizeVisualEnvironment) {
      perAgentStepAndObserve(0, action, reward, done, observation);
      if constexpr (StaticSchedule) {
#pragma omp parallel for schedule(static)
        for (int i = 1; i < num_envs_; i++)
          perAgentStepAndObserve(i, action, reward, done, observation);
      } else {
#pragma omp parallel for schedule(auto)
        for (int i = 1; i < num_envs_; i++)
          perAgentStepAndObserve(i, action, reward, done, observation);
      }
    } else if constexpr (StaticSchedule) {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < num_envs_; i++)
        perAgentStepAndObserve(i, action, reward, done, observation);
    } else {
#pragma omp parallel for schedule(auto)
      for (int i = 0; i < num_envs_; i++)
        perAgentStepAndObserve(i, action, reward, done, observation);
    }
    captureVideoFrame();

    if (normalizeObservation_)
      updateObservationStatisticsAndNormalize(observation, updateStatistics);
  }

  void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
  void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
  /// whether a viewer is connected to the first environment's server
  bool isVisualizerConnected() const { return render_ && environments_[0]->isVisualizerConnected(); }

  /// Record environment 0 into `path` (an .mp4) with an in-process rayrai renderer until
  /// stopRecordingVideo(); no viewer is needed. Returns false, with the reason in
  /// getVideoStatus(), when no OpenGL context or ffmpeg is available. Call the recording
  /// methods and step() from the same thread.
  bool startRecordingVideo(const std::string& path) {
#ifdef RAISIMGYM_WITH_RAYRAI
    if (!videoRecorder_) videoRecorder_ = std::make_unique<VideoRecorder>();
    return videoRecorder_->start(*environments_[0]->getWorld(), path);
#else
    (void)path;
    return false;
#endif
  }

  /// Finish the video. Returns true when the file is complete.
  bool stopRecordingVideo() {
#ifdef RAISIMGYM_WITH_RAYRAI
    return videoRecorder_ && videoRecorder_->stop();
#else
    return false;
#endif
  }

  /// What happened to the last recording, or why it could not start.
  std::string getVideoStatus() const {
#ifdef RAISIMGYM_WITH_RAYRAI
    return videoRecorder_ ? videoRecorder_->status() : std::string("no video recorded yet");
#else
    return "this environment was built without rayrai";
#endif
  }

  void getObStatistics(Eigen::Ref<EigenVec> mean, Eigen::Ref<EigenVec> var, float &count) {
    mean = obMean_; var = obVar_; count = obCount_; }
  void setObStatistics(Eigen::Ref<EigenVec> mean, Eigen::Ref<EigenVec> var, float count) {
    obMean_ = mean; obVar_ = var; obCount_ = count; }

  void setSeed(int seed) {
    // Validate the complete range before invoking any user hook.
    if (int64_t(seed) + num_envs_ - 1 > std::numeric_limits<int>::max())
      throw std::invalid_argument("environment seed range exceeds int");

    if constexpr (StaticSchedule) {
      #pragma omp parallel for schedule(static)
      for(int i=0; i<num_envs_; i++)
        seedEnvironment(i, seed);
    } else {
      #pragma omp parallel for schedule(auto)
      for(int i=0; i<num_envs_; i++)
        seedEnvironment(i, seed);
    }
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec> terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  int getObDim() { return obDim_; }
  int getCriticObDim() { return criticObDim_; }
  int getActionDim() { return actionDim_; }
  int getNumOfEnvs() { return num_envs_; }
  int getNumThreads() const { return THREAD_COUNT; }
  void setNumThreads(int threadCount) {
    if (threadCount < 1)
      throw std::runtime_error("thread count must be positive");

    /// The calling thread is not idle between parallel regions: it runs policy
    /// inference and waits on the accelerator. Handing every hardware thread to
    /// the worker pool makes it compete with the pool, which leaves throughput
    /// unchanged at best and roughly doubles the per-step tail latency.
    const int hardwareThreads = int(std::thread::hardware_concurrency());
    if (hardwareThreads > 1 && threadCount >= hardwareThreads) {
      RSWARN("num_threads=" << threadCount << " requests every one of the "
             << hardwareThreads << " hardware threads. Using "
             << hardwareThreads - 1 << " instead to leave one for the caller.")
      threadCount = hardwareThreads - 1;
    }

    THREAD_COUNT = threadCount;
    omp_set_num_threads(THREAD_COUNT);
  }

  ////// optional methods //////
  void curriculumUpdate() {
    for (auto *env: environments_)
      env->curriculumUpdate();
  };

  const EigenRowMajorMat& getRewardInfo() const { return rewardInformation_; }
  const std::vector<std::string>& getRewardNames() const { return rewardNames_; }

 private:
  void captureVideoFrame() {
#ifdef RAISIMGYM_WITH_RAYRAI
    if (videoRecorder_) videoRecorder_->capture();
#endif
  }

  void seedEnvironment(int index, int seed) {
    environments_[index]->setSeed(seed + index);
    // Keep this outside the virtual hook: derived environments need not call the base.
    // Unsigned conversion preserves negative user seeds without producing a zero World seed.
    environments_[index]->getWorld()->setSensorNoiseSeed(
        uint64_t(uint32_t(seed)) + uint64_t(index) + 1);
  }

  inline void perAgentStepAndObserve(int agentId,
                                     Eigen::Ref<EigenRowMajorMat> &action,
                                     Eigen::Ref<EigenVec> &reward,
                                     Eigen::Ref<EigenBoolVec> &done,
                                     Eigen::Ref<EigenRowMajorMat> &observation) {
    perAgentStep(agentId, action, reward, done);
    environments_[agentId]->observe(observation.row(agentId));
  }

  void updateObservationStatisticsAndNormalize(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics) {
    if (updateStatistics) {
      recentMean_ = ob.colwise().mean();
      recentVar_ = (ob.rowwise() - recentMean_.transpose()).colwise().squaredNorm() / num_envs_;

      delta_ = obMean_ - recentMean_;
      for(int i=0; i<obDim_; i++)
        delta_[i] = delta_[i]*delta_[i];

      float totCount = obCount_ + num_envs_;

      obMean_ = obMean_ * (obCount_ / totCount) + recentMean_ * (num_envs_ / totCount);
      obVar_ = (obVar_ * obCount_ + recentVar_ * num_envs_ + delta_ * (obCount_ * num_envs_ / totCount)) / (totCount);
      obCount_ = totCount;
    }

    /// A few thousand flops: waking the worker team costs more than the work.
    /// Keep the square root inside the row expression. Eigen's vectorized
    /// square root is approximate, and each row's alignment decides which of
    /// its elements take that path, so hoisting it would change the result.
    for(int i=0; i<num_envs_; i++)
      ob.row(i) = (ob.row(i) - obMean_.transpose()).template cwiseQuotient<>((obVar_ + epsilon).cwiseSqrt().transpose());
  }

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                           Eigen::Ref<EigenVec> &reward,
                           Eigen::Ref<EigenBoolVec> &done) {
    reward[agentId] = environments_[agentId]->step(action.row(agentId));
    environments_[agentId]->getRewards().fillRewardValues(rewardInformation_.row(agentId).data(), rewardDim_);

    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    if (done[agentId]) {
      environments_[agentId]->reset();
      reward[agentId] += terminalReward;
    }
  }

  std::vector<ChildEnvironment *> environments_;
#ifdef RAISIMGYM_WITH_RAYRAI
  std::unique_ptr<VideoRecorder> videoRecorder_;
#endif
  EigenRowMajorMat rewardInformation_;
  std::vector<std::string> rewardNames_;

  int num_envs_ = 1;
  int rewardDim_ = 0;
  int obDim_ = 0, criticObDim_ = 0, actionDim_ = 0;
  bool recordVideo_=false, render_=false;
  std::string resourceDir_;
  Yaml::Node cfg_;
  std::string cfgString_;

  /// observation running mean
  bool normalizeObservation_ = true;
  EigenVec obMean_;
  EigenVec obVar_;
  float obCount_ = 1e-4;
  EigenVec recentMean_, recentVar_, delta_;
  EigenVec epsilon;
};


class NormalDistribution {
 public:
  NormalDistribution() : normDist_(0.f, 1.f) {}

  float sample() { return normDist_(gen_); }
  void seed(int i) { gen_.seed(i); }

 private:
  std::normal_distribution<float> normDist_;
  static thread_local std::mt19937 gen_;
};
thread_local std::mt19937 raisim::NormalDistribution::gen_;


class NormalSampler {
 public:
  NormalSampler(int dim) {
    dim_ = dim;
    normal_.resize(THREAD_COUNT);
    seed(0);
  }

  void seed(int seed) {
    // this ensures that every thread gets a different seed
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < THREAD_COUNT; i++)
      normal_[0].seed(i + seed);
  }

  inline void sample(Eigen::Ref<EigenRowMajorMat> mean,
                     Eigen::Ref<EigenVec> std,
                     Eigen::Ref<EigenRowMajorMat> samples,
                     Eigen::Ref<EigenVec> log_prob) {
    int agentNumber = log_prob.rows();

#pragma omp parallel for schedule(auto)
    for (int agentId = 0; agentId < agentNumber; agentId++) {
      log_prob(agentId) = 0;
      for (int i = 0; i < dim_; i++) {
        const float noise = normal_[omp_get_thread_num()].sample();
        samples(agentId, i) = mean(agentId, i) + noise * std(i);
        log_prob(agentId) -= noise * noise * 0.5 + std::log(std(i));
      }
      log_prob(agentId) -= float(dim_) * 0.9189385332f;
    }
  }
  int dim_;
  std::vector<NormalDistribution> normal_;
};

}

#endif //SRC_RAISIMGYMVECENV_HPP
