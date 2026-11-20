//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <tuple>
#include <utility>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

/// ANYmal locomotion on a heightmap that every environment shares.
///
/// Each environment still owns its World, so the environments step in parallel as
/// usual. They share the terrain: a HeightMapResource holds the height samples and the
/// collision tiles, and the HeightMap each World creates from it only references them.
/// The terrain is generated once for all environments and occupies the memory of one
/// heightmap, and adding it to 100 worlds takes well under a millisecond. Likewise, the
/// ANYmal URDF is parsed once into an ArticulatedSystemResource; each World creates its
/// own robot, with its own state, from it.
///
/// The runner calls curriculumUpdate() once per training iteration. Every
/// terrain/randomize_every_n calls, each environment replaces its heightmap with a new
/// shared one and resets its robot onto it.
class ENVIRONMENT : public RaisimGymEnv {

 public:
  static constexpr bool kStaticSchedule = true;
  // Physics steps already synchronize with RaisimServer through its mutex.
  static constexpr bool kParallelizeVisualEnvironment = true;

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// create world
    world_ = std::make_unique<raisim::World>();
    world_->setSleepingEnabled(false);

    /// terrain
    const auto& terrainCfg = cfg["terrain"];
    int terrainSeed;
    READ_YAML(int, randomizeEveryN_, terrainCfg["randomize_every_n"])
    READ_YAML(int, terrainSeed, terrainCfg["seed"])
    READ_YAML(double, terrainSize_, terrainCfg["size"])
    READ_YAML(int, terrainSamples_, terrainCfg["samples"])
    frequency_ = readRange(terrainCfg["frequency"]);
    heightScale_ = readRange(terrainCfg["height_scale"]);
    stepSize_ = readRange(terrainCfg["step_size"]);
    RSFATAL_IF(randomizeEveryN_ < 1, "terrain/randomize_every_n must be at least 1")
    RSFATAL_IF(terrainSamples_ < 2, "terrain/samples must be at least 2")
    /// The robot starts in the central half of the terrain and the episode ends near the
    /// edge, which keeps the height scan on the heightmap.
    RSFATAL_IF(terrainSize_ <= 4. * kEdgeMargin,
               "terrain/size must exceed " << 4. * kEdgeMargin << " m")
    terrainSeed_ = uint32_t(terrainSeed);
    spawnHalfWidth_ = 0.25 * terrainSize_;
    edgeLimit_ = 0.5 * terrainSize_ - kEdgeMargin;
    replaceTerrain(terrainProperties(0));

    /// add objects
    anymal_ = world_->addArticulatedSystem(sharedRobot(resourceDir_+"/anymal/urdf/anymal.urdf"),
                                           raisim::CollisionGroup(1), raisim::CollisionGroup(-1));
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_init_.setZero(gcDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal. reset() moves it onto the terrain.
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 33 + kScanPoints;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);
    heightScan_.setZero(kScanPoints);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(anymal_);
    }
  }

  void init() final {
    updateIntegrationSteps();
  }

  void setSimulationTimeStep(double dt) {
    RaisimGymEnv::setSimulationTimeStep(dt);
    updateIntegrationSteps();
  }

  void setControlTimeStep(double dt) {
    RaisimGymEnv::setControlTimeStep(dt);
    updateIntegrationSteps();
  }

  void setSeed(int seed) final {
    gen_.seed(uint32_t(seed));
  }

  void reset() final {
    /// random position and heading
    std::uniform_real_distribution<double> position(-spawnHalfWidth_, spawnHalfWidth_);
    std::uniform_real_distribution<double> heading(-kPi, kPi);
    const double x = position(gen_), y = position(gen_), yaw = heading(gen_);
    const double c = std::cos(yaw), s = std::sin(yaw);

    /// stand on the highest terrain under the feet, so that no foot starts in the ground
    double ground = terrain_->getHeight(x, y);
    for (double fx : {-kFootX, kFootX})
      for (double fy : {-kFootY, kFootY})
        ground = std::max(ground, terrain_->getHeight(x + c * fx - s * fy, y + s * fx + c * fy));

    Eigen::VectorXd gc = gc_init_;
    gc[0] = x;
    gc[1] = y;
    gc[2] += ground;
    gc[3] = std::cos(0.5 * yaw);
    gc[6] = std::sin(0.5 * yaw);
    anymal_->setState(gc, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    anymal_->setPdTarget(pTarget_, vTarget_);

    if (server_) {
      for (int i = 0; i < integrationSteps_; i++) {
        server_->lockVisualizationServerMutex();
        world_->integrate();
        server_->unlockVisualizationServerMutex();
      }
    } else {
      for (int i = 0; i < integrationSteps_; i++)
        world_->integrate();
    }

    updateObservation();

    rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
    rewards_.record("forwardVel", std::min(4.0, bodyLinearVel_[0]));

    return rewards_.sum();
  }

  void updateObservation() {
    const auto& gc = anymal_->getGeneralizedCoordinate();
    const auto& gv = anymal_->getGeneralizedVelocity();
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc[3]; quat[1] = gc[4]; quat[2] = gc[5]; quat[3] = gc[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv.e().segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv.e().segment(3, 3);

    /// base height above the terrain on a grid that turns with the heading only
    const double yaw = std::atan2(rot.e()(1, 0), rot.e()(0, 0));
    const double c = std::cos(yaw), s = std::sin(yaw);
    int point = 0;
    for (double dx : kScanX)
      for (double dy : kScanY)
        heightScan_[point++] = gc[2] - terrain_->getHeight(gc[0] + c * dx - s * dy,
                                                           gc[1] + s * dx + c * dy);

    obDouble_ << rot.e().row(2).transpose(), /// body orientation
        gc.e().tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv.e().tail(12), /// joint velocity
        heightScan_; /// terrain around the robot
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: anymal_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    /// reaching the edge of the terrain ends the episode without the contact penalty
    terminalReward = 0.f;
    const auto& gc = anymal_->getGeneralizedCoordinate();
    return std::abs(gc[0]) > edgeLimit_ || std::abs(gc[1]) > edgeLimit_;
  }

  /// Called once per training iteration by the runner.
  void curriculumUpdate() final {
    if (++curriculumIteration_ % randomizeEveryN_ != 0)
      return;

    replaceTerrain(terrainProperties(uint32_t(curriculumIteration_ / randomizeEveryN_)));
    reset();
  }

 private:
  struct Range { double min, max; };

  static Range readRange(const Yaml::Node& node) {
    Range range;
    READ_YAML(double, range.min, node["min"])
    READ_YAML(double, range.max, node["max"])
    RSFATAL_IF(range.min > range.max, "terrain range has min " << range.min << " > max " << range.max)
    return range;
  }

  /// Parameters of terrain number `index`. The random generator is seeded with the
  /// terrain seed and the index alone, so every environment derives the same terrain.
  TerrainProperties terrainProperties(uint32_t index) const {
    std::seed_seq seeds{terrainSeed_, index};
    std::mt19937 gen(seeds);
    std::uniform_real_distribution<double> unit(0., 1.);
    const auto draw = [&](const Range& range) {
      return range.min + (range.max - range.min) * unit(gen);
    };

    TerrainProperties properties;
    properties.xSize = properties.ySize = terrainSize_;
    properties.xSamples = properties.ySamples = size_t(terrainSamples_);
    properties.fractalOctaves = 4;
    properties.fractalLacunarity = 2.0;
    properties.fractalGain = 0.5;
    properties.frequency = draw(frequency_);
    properties.zScale = draw(heightScale_);
    properties.stepSize = draw(stepSize_);
    properties.seed = gen();
    return properties;
  }

  /// The heightmap resource for these properties, generated only if they differ from the
  /// previous request. Environments ask in turn with identical properties, so the first one
  /// generates the terrain and the others receive the same resource.
  static HeightMapResource sharedTerrain(const TerrainProperties& properties) {
    static std::mutex mutex;
    static std::optional<std::pair<TerrainProperties, HeightMapResource>> latest;
    const auto key = [](const TerrainProperties& p) {
      return std::make_tuple(p.frequency, p.xSize, p.ySize, p.zScale, p.fractalLacunarity,
                             p.fractalGain, p.stepSize, p.heightOffset, p.fractalOctaves,
                             p.xSamples, p.ySamples, p.seed);
    };

    std::lock_guard<std::mutex> lock(mutex);
    if (!latest || key(latest->first) != key(properties)) {
      latest.emplace(properties, HeightMapResource::generate(0., 0., properties));
      std::cout << "[RAISIM_GYM] New shared heightmap: frequency " << properties.frequency
                << " /m, height scale " << properties.zScale << " m, step " << properties.stepSize
                << " m" << std::endl;
    }
    return latest->second;
  }

  /// The robot model, parsed once and instantiated by every environment.
  static ArticulatedSystemResource sharedRobot(const std::string& urdfPath) {
    static std::mutex mutex;
    static std::map<std::string, ArticulatedSystemResource> resources;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = resources.find(urdfPath);
    if (found == resources.end()) {
      ArticulatedSystemOption options;
      options.convexifyCollisionMeshes = true;
      found = resources.emplace(urdfPath, ArticulatedSystemResource::load(urdfPath, "", {}, options)).first;
    }
    return found->second;
  }

  void replaceTerrain(const TerrainProperties& properties) {
    const HeightMapResource resource = sharedTerrain(properties);
    /// the server thread reads the world while a viewer is connected
    if (server_) server_->lockVisualizationServerMutex();
    if (terrain_) world_->removeObject(terrain_);
    terrain_ = world_->addHeightMap(resource);
    if (server_) server_->unlockVisualizationServerMutex();
  }

  void updateIntegrationSteps() {
    integrationSteps_ = int(control_dt_ / simulation_dt_ + 1e-10);
  }

  static constexpr double kPi = 3.14159265358979323846;
  /// half extents of the standing footprint (m)
  static constexpr double kFootX = 0.35, kFootY = 0.2;
  /// height-scan grid in the heading frame (m): 0.4 m behind to 0.8 m ahead, 0.3 m to each side
  static constexpr std::array<double, 7> kScanX = {-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8};
  static constexpr std::array<double, 5> kScanY = {-0.3, -0.15, 0.0, 0.15, 0.3};
  static constexpr int kScanPoints = int(kScanX.size() * kScanY.size());
  /// distance from the terrain edge that ends an episode; larger than the scan reach (m)
  static constexpr double kEdgeMargin = 1.0;

  int gcDim_, gvDim_, nJoints_, integrationSteps_ = 0;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* anymal_;
  raisim::HeightMap* terrain_ = nullptr;
  Eigen::VectorXd gc_init_, gv_init_, pTarget_, pTarget12_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, heightScan_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  std::mt19937 gen_;

  /// terrain configuration
  int randomizeEveryN_ = 10, terrainSamples_ = 401, curriculumIteration_ = 0;
  uint32_t terrainSeed_ = 0;
  double terrainSize_ = 40., spawnHalfWidth_ = 10., edgeLimit_ = 19.;
  Range frequency_{}, heightScale_{}, stepSize_{};
};

}
