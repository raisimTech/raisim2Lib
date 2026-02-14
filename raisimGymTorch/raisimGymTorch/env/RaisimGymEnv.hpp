//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMENV_HPP
#define SRC_RAISIMGYMENV_HPP

#include <vector>
#include <memory>
#include <unordered_map>
#include "Common.hpp"
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Yaml.hpp"
#include "Reward.hpp"

namespace raisim {


class RaisimGymEnv {

 public:
  explicit RaisimGymEnv (std::string resourceDir, const Yaml::Node& cfg) :
      resourceDir_(std::move(resourceDir)), cfg_(cfg) { }

  virtual ~RaisimGymEnv() { if(server_) server_->killServer(); };

  /////// implement these methods /////////
  virtual void init() = 0;
  virtual void reset() = 0;
  virtual void observe(Eigen::Ref<EigenVec> ob) = 0;
  virtual void observeCritic(Eigen::Ref<EigenVec> ob) { observe(ob); }
  virtual float step(const Eigen::Ref<EigenVec>& action) = 0;
  virtual bool isTerminalState(float& terminalReward) = 0;
  ////////////////////////////////////////

  /////// optional methods ///////
  virtual void curriculumUpdate() {};
  virtual void close() {};
  virtual void setSeed(int seed) {};
  ////////////////////////////////

  void setSimulationTimeStep(double dt) { simulation_dt_ = dt; world_->setTimeStep(dt); }
  void setControlTimeStep(double dt) { control_dt_ = dt; }
  int getObDim() const { return obDim_; }
  virtual int getCriticObDim() const { return criticObDim_ > 0 ? criticObDim_ : obDim_; }
  int getActionDim() const { return actionDim_; }
  double getControlTimeStep() const { return control_dt_; }
  double getSimulationTimeStep() const { return simulation_dt_; }
  World* getWorld() const { return world_.get(); }
  virtual void turnOffVisualization() { server_->hibernate(); }
  virtual void turnOnVisualization() { server_->wakeup(); }
  virtual void startRecordingVideo(const std::string& videoName ) { server_->startRecordingVideo(videoName); }
  virtual void stopRecordingVideo() { server_->stopRecordingVideo(); }
  Reward& getRewards() { return rewards_; }

 protected:
  std::unique_ptr<World> world_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  std::string resourceDir_;
  Yaml::Node cfg_;
  int obDim_=0, actionDim_=0;
  int criticObDim_=0;
  std::unique_ptr<RaisimServer> server_;
  Reward rewards_;
};
}

#endif //SRC_RAISIMGYMENV_HPP
