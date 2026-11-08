// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#ifndef RAISIM_EXAMPLES_INCLUDE_ARM_IK_DEMO_HPP_
#define RAISIM_EXAMPLES_INCLUDE_ARM_IK_DEMO_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "raisim/World.hpp"

// The scene, the IK controller and the pointer math of rayrai_arm_ik. The arm
// is the Kinova/Robotiq model of robotiq_pick_and_place.hpp on the same
// workbench, with three loose cubes instead of the scripted one.
namespace raisim_examples::arm_ik {

constexpr double kTimeStep = 0.001;
constexpr double kTableHeight = 0.32;
constexpr double kCubeSize = 0.04;
// Knuckle angle that closes the fingers on a cube.
constexpr double kGripClosed = 0.75;
// The commanded joint angles close their gap to the IK solution with this
// time constant [s], without any joint turning faster than kMaxJointSpeed.
constexpr double kApproachTime = 0.08;
constexpr double kMaxJointSpeed = 2.0;
constexpr double kMaxGripSpeed = 1.5;
// The target stays above the ground and this close to the arm's base [m], so
// a drag towards the horizon cannot carry it out of sight.
constexpr double kMaxTargetDistance = 1.1;
// One lap of the --tour path [s].
constexpr double kTourPeriod = 12.0;
constexpr double kPi = 3.14159265358979323846;
// The IK frame is a point between the fingertips (see kinova_robotiq.urdf).
constexpr const char* kIkFrame = "grasp_frame_joint";

// Gripper pointing straight down, its grasp frame at (0.35, 0, 0.46). Higher
// or further out, a gripper pointing down nears the arm's singular poses, and
// small turns of the hand need large, slow joint motions or none at all.
using ArmPose = std::array<double, 6>;
constexpr ArmPose kHome{-0.0280036599681, 3.0295117382, -1.89610333104, 0.0, 1.35757023794, -0.0280036599555};

/// A pointer ray in world coordinates; `direction` has unit length.
struct Ray {
  Eigen::Vector3d origin;
  Eigen::Vector3d direction;
};

/// Distance between `point` and the ray (to its origin for points behind it).
inline double DistanceToRay(const Ray& ray, const Eigen::Vector3d& point) {
  const double along = std::max(0.0, ray.direction.dot(point - ray.origin));
  return (ray.origin + along * ray.direction - point).norm();
}

/// Moves a target with the pointer. The grabbed point stays under the pointer
/// on the plane through the target that faces the camera.
class TargetDrag {
 public:
  /// Grabs the target if the ray passes within `radius` of it. `view` is the
  /// camera's unit viewing direction. Returns whether the target was grabbed.
  bool Begin(const Ray& ray, const Eigen::Vector3d& target, const Eigen::Vector3d& view, double radius) {
    if (DistanceToRay(ray, target) > radius) return false;
    point_ = target;
    normal_ = view;
    Eigen::Vector3d hit;
    if (!Intersect(ray, hit)) return false;
    offset_ = target - hit;
    active_ = true;
    return true;
  }

  /// Target position for the pointer ray. Returns `target` unchanged when
  /// nothing is grabbed or the ray misses the plane (parallel to it or
  /// pointing away from it).
  Eigen::Vector3d Move(const Ray& ray, const Eigen::Vector3d& target) const {
    Eigen::Vector3d hit;
    if (!active_ || !Intersect(ray, hit)) return target;
    return hit + offset_;
  }

  /// Moves the plane, and with it the grabbed target, along the viewing
  /// direction (away from the camera for positive distances).
  void Push(double distance) { point_ += distance * normal_; }
  void End() { active_ = false; }
  bool active() const { return active_; }

 private:
  bool Intersect(const Ray& ray, Eigen::Vector3d& hit) const {
    const double rate = ray.direction.dot(normal_);
    if (std::abs(rate) < 1e-6) return false;
    const double distance = (point_ - ray.origin).dot(normal_) / rate;
    if (distance <= 0.0) return false;
    hit = ray.origin + distance * ray.direction;
    return true;
  }

  bool active_ = false;
  Eigen::Vector3d point_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_ = Eigen::Vector3d::UnitX();
  Eigen::Vector3d offset_ = Eigen::Vector3d::Zero();
};

/// Turns `orientation` as if the pointer rolled it: moving (dx, dy) pixels
/// (y down) turns it about the camera's up axis and right axis, so its side
/// facing the camera follows the pointer.
inline Eigen::Matrix3d RotateByDrag(const Eigen::Matrix3d& orientation, double dx, double dy, const Eigen::Vector3d& right,
                                    const Eigen::Vector3d& up, double radians_per_pixel) {
  const Eigen::Quaterniond turn = Eigen::AngleAxisd(dy * radians_per_pixel, right.normalized()) *
                                  Eigen::AngleAxisd(dx * radians_per_pixel, up.normalized());
  return (turn * Eigen::Quaterniond(orientation)).normalized().toRotationMatrix();
}

/// The arm on its workbench with three cubes, and the controller that makes
/// the gripper follow a target pose.
///
/// SolveIk() runs ArticulatedSystem::solveIK() from the previous solution
/// without moving the simulated arm. Step() moves the arm's PD targets towards
/// that solution and integrates, so the arm stays a physical body that pushes,
/// grasps and lifts the cubes.
class Scene {
 public:
  explicit Scene(const std::string& model);

  /// Sets the pose the grasp frame follows. The position is kept above the
  /// ground and within kMaxTargetDistance of the base; a non-finite pose is
  /// ignored.
  void SetTarget(const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation);
  /// Follows the target's orientation too, or only its position. Switching it
  /// on starts from the gripper's current orientation, so the hand does not
  /// turn suddenly.
  void SetTrackOrientation(bool track);
  void SetGripperClosed(bool closed) { gripper_closed_ = closed; }
  /// Solves IK for the target. Call when the target or the options changed.
  void SolveIk();
  /// Moves the PD targets towards the IK solution and integrates one step.
  void Step();
  /// Restores the initial state with the target at the home pose and the
  /// gripper open.
  void Reset();
  /// Pose of the --tour path at `time`; it starts and ends at the home pose.
  void TourPose(double time, Eigen::Vector3d& position, Eigen::Matrix3d& orientation) const;
  /// Pose of the grasp frame.
  void GraspPose(Eigen::Vector3d& position, Eigen::Matrix3d& orientation) const;

  raisim::World& world() { return world_; }
  raisim::ArticulatedSystem* robot() const { return robot_; }
  const std::vector<raisim::Box*>& cubes() const { return cubes_; }
  /// Options of the IK solve; Scene sets `seed`, `applySolution` and
  /// `useOrientation` itself.
  raisim::ArticulatedSystem::IKOptions& ik_options() { return ik_options_; }
  const raisim::ArticulatedSystem::IKResult& ik_result() const { return ik_result_; }
  const Eigen::VectorXd& ik_solution() const { return ik_solution_; }
  const Eigen::VectorXd& command() const { return command_; }
  const Eigen::VectorXd& command_velocity() const { return command_velocity_; }
  const Eigen::Vector3d& target_position() const { return target_position_; }
  const Eigen::Matrix3d& target_orientation() const { return target_orientation_; }
  bool track_orientation() const { return track_orientation_; }
  bool gripper_closed() const { return gripper_closed_; }
  std::size_t arm_index(std::size_t joint) const { return arm_indices_[joint]; }
  std::size_t grip_index() const { return grip_index_; }

 private:
  static constexpr std::array<double, 5> kFollowerSigns{-1.0, 1.0, -1.0, -1.0, 1.0};

  /// Commands the home pose with the gripper open and makes it the IK seed.
  void CommandHome();

  raisim::World world_;
  raisim::ArticulatedSystem* robot_ = nullptr;
  std::vector<raisim::Box*> cubes_;
  std::size_t ik_frame_ = 0;
  std::array<std::size_t, 6> arm_indices_{};
  std::array<std::size_t, 5> follower_indices_{};
  std::size_t grip_index_ = 0;
  raisim::ArticulatedSystem::IKOptions ik_options_;
  raisim::ArticulatedSystem::IKResult ik_result_;
  Eigen::VectorXd ik_solution_;
  Eigen::VectorXd command_;
  Eigen::VectorXd command_velocity_;
  double grip_ = 0.0;
  bool gripper_closed_ = false;
  bool track_orientation_ = true;
  Eigen::Vector3d home_position_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d home_orientation_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d target_position_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d target_orientation_ = Eigen::Matrix3d::Identity();
  raisim::World::Checkpoint checkpoint_;
};

inline Scene::Scene(const std::string& model) {
  world_.setTimeStep(kTimeStep);
  // A sleeping arm would not wake up for a new PD target.
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

  raisim::ArticulatedSystemOption options;
  options.convexifyCollisionMeshes = true;
  robot_ = world_.addArticulatedSystem(model, "", {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);
  robot_->setName("kinova_robotiq");
  if (robot_->getDOF() != 12 || robot_->getGeneralizedCoordinateDim() != 12) {
    throw std::runtime_error("The fixed-base Kinova/Robotiq model must have twelve joints");
  }
  ik_frame_ = robot_->getFrameIdxByName(kIkFrame);
  for (std::size_t i = 0; i < arm_indices_.size(); ++i) {
    arm_indices_[i] = robot_->getGeneralizedVelocityIndex("kinova_joint_" + std::to_string(i + 1));
  }
  grip_index_ = robot_->getGeneralizedVelocityIndex("robotiq_85_left_knuckle_joint");
  const std::array<const char*, 5> followers{"robotiq_85_right_knuckle_joint", "robotiq_85_left_inner_knuckle_joint", "robotiq_85_right_inner_knuckle_joint", "robotiq_85_left_finger_tip_joint",
                                             "robotiq_85_right_finger_tip_joint"};
  for (std::size_t i = 0; i < followers.size(); ++i) {
    follower_indices_[i] = robot_->getGeneralizedVelocityIndex(followers[i]);
  }
  const std::size_t left_pad = robot_->getBodyIdx("robotiq_85_left_finger_tip_link");
  const std::size_t right_pad = robot_->getBodyIdx("robotiq_85_right_finger_tip_link");
  for (auto& body : robot_->getCollisionBodies()) {
    body.setMaterial(body.localIdx == left_pad || body.localIdx == right_pad ? "pad" : "arm");
  }

  Eigen::VectorXd p_gain = Eigen::VectorXd::Zero(robot_->getDOF());
  Eigen::VectorXd d_gain = p_gain;
  const ArmPose p{2000.0, 3000.0, 2500.0, 600.0, 600.0, 300.0};
  const ArmPose d{60.0, 80.0, 70.0, 20.0, 20.0, 12.0};
  for (std::size_t i = 0; i < arm_indices_.size(); ++i) {
    p_gain[arm_indices_[i]] = p[i];
    d_gain[arm_indices_[i]] = d[i];
  }
  // Mimic followers stay passive, as in robotiq_pick_and_place.hpp.
  p_gain[grip_index_] = 10.0;
  d_gain[grip_index_] = 0.2;
  robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  robot_->setPdGains(p_gain, d_gain);
  robot_->setGeneralizedForce(Eigen::VectorXd::Zero(robot_->getDOF()));

  const std::array<Eigen::Vector2d, 3> spots{Eigen::Vector2d(0.44, -0.16), Eigen::Vector2d(0.52, 0.0), Eigen::Vector2d(0.44, 0.16)};
  const std::array<const char*, 3> colors{"0.85,0.22,0.22,1", "0.22,0.68,0.32,1", "0.22,0.42,0.88,1"};
  for (std::size_t i = 0; i < spots.size(); ++i) {
    auto* cube = world_.addBox(kCubeSize, kCubeSize, kCubeSize, 0.1, "cube");
    cube->setName("cube_" + std::to_string(i));
    cube->setPosition(spots[i].x(), spots[i].y(), kTableHeight + 0.5 * kCubeSize);
    cube->setAppearance(colors[i]);
    cubes_.push_back(cube);
  }
  world_.setMaterialPairProp("pad", "cube", 1.0, 0.0, 0.001);
  world_.setMaterialPairProp("table", "cube", 0.8, 0.0, 0.001);
  world_.setMaterialPairProp("arm", "cube", 0.5, 0.0, 0.001);

  ik_options_.maxIterations = 50;
  ik_options_.orientationTolerance = 1.0e-3;
  // Trade 1 rad of hand orientation for 0.3 m of position when the target is
  // out of reach.
  ik_options_.orientationWeight = 0.3;

  CommandHome();
  robot_->setGeneralizedCoordinate(command_);
  robot_->setGeneralizedVelocity(command_velocity_);
  GraspPose(home_position_, home_orientation_);
  target_position_ = home_position_;
  target_orientation_ = home_orientation_;
  // Let gravity settle the cubes, then keep this state for Reset().
  for (int i = 0; i < 500; ++i) Step();
  checkpoint_ = world_.captureCheckpoint();
}

inline void Scene::CommandHome() {
  command_ = Eigen::VectorXd::Zero(robot_->getDOF());
  command_velocity_ = command_;
  for (std::size_t i = 0; i < arm_indices_.size(); ++i) command_[arm_indices_[i]] = kHome[i];
  robot_->setPdTarget(command_, command_velocity_);
  ik_solution_ = command_;
  ik_result_ = raisim::ArticulatedSystem::IKResult();
  ik_result_.success = true;
  grip_ = 0.0;
  gripper_closed_ = false;
}

inline void Scene::SetTarget(const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation) {
  if (!position.allFinite() || !orientation.allFinite()) return;
  target_position_ = position;
  target_position_.z() = std::max(target_position_.z(), 0.0);
  const double distance = target_position_.norm();
  if (distance > kMaxTargetDistance) target_position_ *= kMaxTargetDistance / distance;
  target_orientation_ = orientation;
}

inline void Scene::SetTrackOrientation(bool track) {
  if (track && !track_orientation_) {
    Eigen::Vector3d position;
    GraspPose(position, target_orientation_);
  }
  track_orientation_ = track;
}

inline void Scene::SolveIk() {
  // Starting from the previous solution keeps the arm on one IK branch, and
  // leaving the simulated state alone lets Step() move it physically.
  ik_options_.seed.assign(ik_solution_.data(), ik_solution_.data() + ik_solution_.size());
  ik_options_.applySolution = false;
  ik_options_.useOrientation = track_orientation_;
  raisim::Vec<3> position;
  raisim::Mat<3, 3> orientation;
  position.e() = target_position_;
  orientation.e() = target_orientation_;
  ik_result_ = robot_->solveIK(ik_frame_, position, orientation, ik_options_);
  // Out of reach, the closest pose found is the best the arm can do.
  const auto& solution = ik_result_.success ? ik_result_.solution : ik_result_.bestSolution;
  if (solution.size() == static_cast<std::size_t>(ik_solution_.size())) {
    ik_solution_ = Eigen::Map<const Eigen::VectorXd>(solution.data(), ik_solution_.size());
  }
}

inline void Scene::Step() {
  // Close the gap to the solution with time constant kApproachTime, scaling
  // all joints together so the hand moves along a joint-space straight line.
  Eigen::Matrix<double, 6, 1> speed;
  for (std::size_t i = 0; i < arm_indices_.size(); ++i) {
    speed[i] = (ik_solution_[arm_indices_[i]] - command_[arm_indices_[i]]) / kApproachTime;
  }
  const double fastest = speed.cwiseAbs().maxCoeff();
  if (fastest > kMaxJointSpeed) speed *= kMaxJointSpeed / fastest;
  for (std::size_t i = 0; i < arm_indices_.size(); ++i) {
    command_[arm_indices_[i]] += speed[i] * kTimeStep;
    command_velocity_[arm_indices_[i]] = speed[i];
  }
  const double grip_goal = gripper_closed_ ? kGripClosed : 0.0;
  const double grip_speed = std::clamp((grip_goal - grip_) / kApproachTime, -kMaxGripSpeed, kMaxGripSpeed);
  grip_ += grip_speed * kTimeStep;
  command_[grip_index_] = grip_;
  command_velocity_[grip_index_] = grip_speed;
  // The followers' targets only document the linkage; they have no gains.
  for (std::size_t i = 0; i < follower_indices_.size(); ++i) {
    command_[follower_indices_[i]] = kFollowerSigns[i] * grip_;
    command_velocity_[follower_indices_[i]] = kFollowerSigns[i] * grip_speed;
  }
  robot_->setPdTarget(command_, command_velocity_);
  world_.integrate();
}

inline void Scene::Reset() {
  std::string error;
  if (!world_.restoreCheckpoint(checkpoint_, &error)) {
    throw std::runtime_error("Cannot reset the IK scene: " + error);
  }
  CommandHome();
  target_position_ = home_position_;
  target_orientation_ = home_orientation_;
}

inline void Scene::TourPose(double time, Eigen::Vector3d& position, Eigen::Matrix3d& orientation) const {
  // A figure eight that dips towards the table beside the cubes, with the
  // hand turning about the vertical and tilting.
  const double phase = 2.0 * kPi * time / kTourPeriod;
  position = home_position_ + Eigen::Vector3d(0.08 * std::sin(2.0 * phase), 0.18 * std::sin(phase), 0.03 * (std::cos(2.0 * phase) - 1.0));
  orientation = (Eigen::AngleAxisd(0.6 * std::sin(phase), Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(0.3 * std::sin(2.0 * phase), Eigen::Vector3d::UnitY())).toRotationMatrix() *
                home_orientation_;
}

inline void Scene::GraspPose(Eigen::Vector3d& position, Eigen::Matrix3d& orientation) const {
  raisim::Vec<3> p;
  raisim::Mat<3, 3> r;
  robot_->getFramePosition(ik_frame_, p);
  robot_->getFrameOrientation(ik_frame_, r);
  position = p.e();
  orientation = r.e();
}

}  // namespace raisim_examples::arm_ik

#endif  // RAISIM_EXAMPLES_INCLUDE_ARM_IK_DEMO_HPP_
