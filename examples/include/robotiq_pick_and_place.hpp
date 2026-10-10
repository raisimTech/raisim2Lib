// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#ifndef RAISIM_EXAMPLES_INCLUDE_ROBOTIQ_PICK_AND_PLACE_HPP_
#define RAISIM_EXAMPLES_INCLUDE_ROBOTIQ_PICK_AND_PLACE_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "Eigen/Core"
#include "raisim/World.hpp"

namespace raisim_examples::robotiq {

constexpr double kTimeStep = 0.001;
constexpr double kTableHeight = 0.32;
constexpr double kCubeSize = 0.04;
constexpr double kCubeMass = 0.10;
constexpr double kClosed = 0.75;
inline const Eigen::Vector3d kPickup{0.42, -0.18,
                                     kTableHeight + 0.5 * kCubeSize};
inline const Eigen::Vector3d kDestination{0.42, 0.18,
                                          kTableHeight + 0.5 * kCubeSize};

// Offline joint poses for a downward-facing grasp_frame at z=0.54 / z=0.34.
// Runtime motion is entirely PD control; no IK, base motion, or cube
// attachment.
using ArmPose = std::array<double, 6>;
constexpr ArmPose kPickupHover{0.38344342446, 2.53791743213, -2.94081775313,
                               0.0,           0.80445012192, 0.38344342446};
constexpr ArmPose kPickupGrasp{0.38344342446, 2.77440756619, -1.84325682091,
                               0.0,           1.66552092008, 0.38344342446};
constexpr ArmPose kDestinationHover{-0.42634014811, 2.53791743213,
                                    -2.94081775313, 0.0,
                                    0.80445012192,  -0.42634014811};
constexpr ArmPose kDestinationGrasp{-0.42634014811, 2.77440756619,
                                    -1.84325682091, 0.0,
                                    1.66552092008,  -0.42634014811};

struct Phase {
  const char* name;
  double duration;
  ArmPose from;
  ArmPose to;
  double grip_from;
  double grip_to;
};

inline constexpr std::array<Phase, 10> kPhases{{
    {"Ready", 1.0, kPickupHover, kPickupHover, 0.0, 0.0},
    {"Approach pickup", 2.5, kPickupHover, kPickupGrasp, 0.0, 0.0},
    {"Grasp", 1.5, kPickupGrasp, kPickupGrasp, 0.0, kClosed},
    {"Lift", 2.5, kPickupGrasp, kPickupHover, kClosed, kClosed},
    {"Transfer", 3.5, kPickupHover, kDestinationHover, kClosed, kClosed},
    {"Lower", 2.5, kDestinationHover, kDestinationGrasp, kClosed, kClosed},
    {"Release", 1.5, kDestinationGrasp, kDestinationGrasp, kClosed, 0.0},
    {"Retract", 2.5, kDestinationGrasp, kDestinationHover, 0.0, 0.0},
    {"Return", 3.5, kDestinationHover, kPickupHover, 0.0, 0.0},
    {"Reset", 0.5, kPickupHover, kPickupHover, 0.0, 0.0},
}};

inline constexpr double kCycleDuration = [] {
  double duration = 0.0;
  for (const auto& phase : kPhases) {
    duration += phase.duration;
  }
  return duration;
}();

struct CycleResult {
  bool lifted = false;
  bool carried = false;
  bool released = false;
  bool bilateral_grasp = false;
  double max_cube_height = 0.0;
  double max_mimic_error = 0.0;
  double max_arm_error = 0.0;
  Eigen::Vector3d placed_position = Eigen::Vector3d::Zero();

  bool IsSuccessful() const {
    return lifted && carried && released && bilateral_grasp;
  }
};

class Scene {
 public:
  explicit Scene(const std::string& model);

  // Invoke under RaisimServer's world mutex immediately before integration.
  // Simulation time drives the script, so viewer pause and single-step work.
  void Update();
  void Step() {
    Update();
    world_.integrate();
  }

  raisim::World& world() { return world_; }
  raisim::ArticulatedSystem* robot() const { return robot_; }
  raisim::Box* cube() const { return cube_; }
  std::size_t phase_index() const { return phase_index_; }
  const CycleResult& last_result() const { return last_result_; }
  std::size_t completed_cycles() const { return completed_cycles_; }

 private:
  static constexpr std::array<double, 5> kFollowerSigns{-1.0, 1.0, -1.0, -1.0,
                                                        1.0};

  void Observe();

  raisim::World world_;
  raisim::ArticulatedSystem* robot_ = nullptr;
  raisim::Box* cube_ = nullptr;
  std::size_t phase_index_ = 0;
  CycleResult current_;
  CycleResult last_result_;
  std::size_t completed_cycles_ = 0;
  std::array<std::size_t, 6> arm_indices_{};
  std::array<std::size_t, 5> follower_indices_{};
  std::size_t grip_index_ = 0;
  std::size_t left_pad_ = 0;
  std::size_t right_pad_ = 0;
  Eigen::VectorXd target_;
  Eigen::VectorXd velocity_target_;
  raisim::World::Checkpoint checkpoint_;
  double start_time_ = 0.0;
};

inline Scene::Scene(const std::string& model) {
  world_.setTimeStep(kTimeStep);
  world_.setSleepingEnabled(false);
  world_.addGround(0.0, "ground")->setAppearance("checkerboard");

  auto* table = world_.addBox(0.58, 0.74, 0.04, 1.0, "table");
  table->setName("workbench");
  table->setBodyType(raisim::BodyType::STATIC);
  table->setPosition(0.43, 0.0, kTableHeight - 0.02);
  table->setAppearance("0.32,0.36,0.42,1");
  for (double x : {0.19, 0.67}) {
    for (double y : {-0.31, 0.31}) {
      auto* leg = world_.addBox(0.025, 0.025, kTableHeight - 0.04, 1.0);
      leg->setBodyType(raisim::BodyType::STATIC);
      leg->setPosition(x, y, 0.5 * (kTableHeight - 0.04));
      leg->setAppearance("0.16,0.18,0.21,1");
    }
  }
  // Raise the collision-free marker tops 1 mm to avoid coplanar depth fighting.
  for (const auto& position : {kPickup, kDestination}) {
    auto* marker = world_.addBox(0.10, 0.10, 0.001, 1.0, "table", 0, 0);
    marker->setBodyType(raisim::BodyType::STATIC);
    marker->setPosition(position[0], position[1], kTableHeight + 0.0005);
    marker->setAppearance(position[1] < 0.0 ? "0.15,0.55,0.90,1"
                                            : "0.25,0.75,0.35,1");
  }

  robot_ = world_.addArticulatedSystem(model);
  robot_->setName("kinova_robotiq");
  if (robot_->getDOF() != 12 || robot_->getGeneralizedCoordinateDim() != 12) {
    throw std::runtime_error(
        "The fixed-base Kinova/Robotiq model must have twelve joints");
  }
  for (std::size_t i = 0; i < arm_indices_.size(); ++i) {
    arm_indices_[i] = robot_->getGeneralizedVelocityIndex(
        "kinova_joint_" + std::to_string(i + 1));
  }
  grip_index_ =
      robot_->getGeneralizedVelocityIndex("robotiq_85_left_knuckle_joint");
  left_pad_ = robot_->getBodyIdx("robotiq_85_left_finger_tip_link");
  right_pad_ = robot_->getBodyIdx("robotiq_85_right_finger_tip_link");
  for (auto& body : robot_->getCollisionBodies()) {
    body.setMaterial(body.localIdx == left_pad_ || body.localIdx == right_pad_
                         ? "pad"
                         : "arm");
  }
  const std::array<const char*, 5> followers{
      "robotiq_85_right_knuckle_joint", "robotiq_85_left_inner_knuckle_joint",
      "robotiq_85_right_inner_knuckle_joint",
      "robotiq_85_left_finger_tip_joint", "robotiq_85_right_finger_tip_joint"};
  for (std::size_t i = 0; i < followers.size(); ++i) {
    follower_indices_[i] = robot_->getGeneralizedVelocityIndex(followers[i]);
  }

  target_ = Eigen::VectorXd::Zero(robot_->getDOF());
  velocity_target_ = target_;
  Eigen::VectorXd p_gain = target_;
  Eigen::VectorXd d_gain = target_;
  const ArmPose p{2000.0, 3000.0, 2500.0, 600.0, 600.0, 300.0};
  const ArmPose d{60.0, 80.0, 70.0, 20.0, 20.0, 12.0};
  for (std::size_t i = 0; i < arm_indices_.size(); ++i) {
    p_gain[arm_indices_[i]] = p[i];
    d_gain[arm_indices_[i]] = d[i];
    target_[arm_indices_[i]] = kPickupHover[i];
  }
  p_gain[grip_index_] = 10.0;
  d_gain[grip_index_] = 0.2;
  robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  robot_->setGeneralizedCoordinate(target_);
  robot_->setGeneralizedVelocity(velocity_target_);
  robot_->setGeneralizedForce(Eigen::VectorXd::Zero(robot_->getDOF()));
  robot_->setPdGains(p_gain, d_gain);
  robot_->setPdTarget(target_, velocity_target_);

  cube_ = world_.addBox(kCubeSize, kCubeSize, kCubeSize, kCubeMass, "cube");
  cube_->setName("pick_and_place_cube");
  cube_->setPosition(kPickup);
  cube_->setAppearance("0.95,0.55,0.15,1");
  world_.setMaterialPairProp("pad", "cube", 1.0, 0.0, 0.001);
  world_.setMaterialPairProp("table", "cube", 0.8, 0.0, 0.001);
  world_.setMaterialPairProp("arm", "cube", 0.5, 0.0, 0.001);

  // Settle gravity and the support contacts once; repeat the complete physical
  // initial state, including solver history, only at the end of each cycle.
  for (int i = 0; i < 500; ++i) {
    world_.integrate();
  }
  checkpoint_ = world_.captureCheckpoint();
  start_time_ = world_.getWorldTime();
}

inline void Scene::Update() {
  Observe();
  double time = world_.getWorldTime() - start_time_;
  if (time >= kCycleDuration - 0.5 * kTimeStep) {
    last_result_ = current_;
    ++completed_cycles_;
    std::string error;
    if (!world_.restoreCheckpoint(checkpoint_, &error)) {
      throw std::runtime_error("Cannot reset pick-and-place world: " + error);
    }
    start_time_ = world_.getWorldTime();
    current_ = CycleResult{};
    time = 0.0;
  }
  double begin = 0.0;
  for (std::size_t i = 0; i < kPhases.size(); ++i) {
    const auto& phase = kPhases[i];
    if (time < begin + phase.duration || i + 1 == kPhases.size()) {
      phase_index_ = i;
      const double s = std::clamp((time - begin) / phase.duration, 0.0, 1.0);
      const double blend = s * s * s * (10.0 + s * (-15.0 + 6.0 * s));
      const double rate = 30.0 * s * s * (1.0 - s) * (1.0 - s) / phase.duration;
      for (std::size_t j = 0; j < arm_indices_.size(); ++j) {
        const double delta = phase.to[j] - phase.from[j];
        target_[arm_indices_[j]] = phase.from[j] + blend * delta;
        velocity_target_[arm_indices_[j]] = rate * delta;
      }
      const double grip =
          phase.grip_from + blend * (phase.grip_to - phase.grip_from);
      target_[grip_index_] = grip;
      velocity_target_[grip_index_] = rate * (phase.grip_to - phase.grip_from);
      // Mimic followers have zero PD gains; their targets document the same
      // linkage relation and never provide extra squeeze torques.
      for (std::size_t j = 0; j < follower_indices_.size(); ++j) {
        const double sign = kFollowerSigns[j];
        target_[follower_indices_[j]] = sign * grip;
        velocity_target_[follower_indices_[j]] =
            sign * velocity_target_[grip_index_];
      }
      robot_->setPdTarget(target_, velocity_target_);
      break;
    }
    begin += phase.duration;
  }
}

inline void Scene::Observe() {
  const auto& q = robot_->getGeneralizedCoordinate();
  if (!q.e().allFinite() || !robot_->getGeneralizedVelocity().e().allFinite() ||
      !cube_->getPosition().allFinite() ||
      !cube_->getLinearVelocity().allFinite() ||
      !cube_->getAngularVelocity().allFinite()) {
    throw std::runtime_error("Non-finite pick-and-place state");
  }
  for (std::size_t j = 0; j < follower_indices_.size(); ++j) {
    current_.max_mimic_error = std::max(
        current_.max_mimic_error,
        std::abs(q[follower_indices_[j]] - kFollowerSigns[j] * q[grip_index_]));
  }
  for (std::size_t j = 0; j < arm_indices_.size(); ++j) {
    current_.max_arm_error =
        std::max(current_.max_arm_error,
                 std::abs(q[arm_indices_[j]] - target_[arm_indices_[j]]));
  }
  const auto position = cube_->getPosition();
  current_.max_cube_height = std::max(current_.max_cube_height, position[2]);
  bool left = false;
  bool right = false;
  for (const auto& contact : robot_->getContacts()) {
    if (contact.skip() ||
        contact.getPairObjectIndex() != cube_->getIndexInWorld()) {
      continue;
    }
    left |= contact.getlocalBodyIndex() == left_pad_;
    right |= contact.getlocalBodyIndex() == right_pad_;
  }
  if (phase_index_ == 3 || phase_index_ == 4) {
    current_.bilateral_grasp |= left && right;
    current_.lifted |= position[2] > kPickup[2] + 0.15;
  }
  if (phase_index_ == 4) {
    current_.carried |= position[2] > kPickup[2] + 0.15 &&
                        position[1] > kDestination[1] - 0.025;
  }
  if (phase_index_ >= 8) {
    current_.placed_position = position;
    current_.released = (position - kDestination).norm() < 0.015 &&
                        cube_->getLinearVelocity().norm() < 0.02 &&
                        q[grip_index_] < 0.03 && !left && !right;
  }
}

}  // namespace raisim_examples::robotiq

#endif  // RAISIM_EXAMPLES_INCLUDE_ROBOTIQ_PICK_AND_PLACE_HPP_
