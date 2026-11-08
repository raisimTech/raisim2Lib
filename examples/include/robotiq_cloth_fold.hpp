// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#ifndef RAISIM_EXAMPLES_INCLUDE_ROBOTIQ_CLOTH_FOLD_HPP_
#define RAISIM_EXAMPLES_INCLUDE_ROBOTIQ_CLOTH_FOLD_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "raisim/World.hpp"

namespace raisim_examples::cloth_fold {

constexpr double kTimeStep = 0.001;
constexpr double kTableHeight = 0.20;

// The pads start one particle diameter apart around the corner. The knuckle
// target lies further closed, so the PD error squeezes the cloth.
constexpr double kInitialGrip = 0.69;
constexpr double kClosed = 0.75;

// A 0.2 m square cloth hangs by one corner between the pads, which close
// along y, so it is a diamond in a vertical xz-plane. The particle radius is
// large enough that no gap between neighbors admits a 4 mm pad.
constexpr int kClothResolution = 19;
constexpr double kClothSide = 0.20;
constexpr double kClothRadius = 0.006;
inline const double kHalfDiagonal = kClothSide * std::sqrt(0.5);

// grasp_frame waypoints at x=0.42, gripper pointing down with its fingers
// closing along y. The arm starts with the cloth hanging 2 cm above the
// table. It then moves toward -y, across the cloth, while descending, which
// lays the lower half on the table; the fold line ends up near y=0.077.
inline const Eigen::Vector3d kStartPosition{0.42, 0.16, 0.50};
inline const Eigen::Vector3d kLayMidPosition{0.42, 0.12, 0.415};
inline const Eigen::Vector3d kFoldTopPosition{0.42, 0.077, 0.300};

// The fold: the pads' lower edges (kPadEdgeDepth below grasp_frame) circle the
// measured fold line at kFoldRadius. That leaves slack in the 0.11 m of cloth
// between them and the fold, because a taut upper half would drag the lower
// half along the table. The pads stay kMinGraspHeight - kPadEdgeDepth above
// the tabletop, clear of both layers of cloth.
constexpr double kPadEdgeDepth = 0.022;
constexpr double kFoldRadius = 0.08;
constexpr std::array<double, 3> kFoldAngles{0.52359877560, 0.95993108860, 1.39626340160};  // 30, 55, 80 deg
constexpr double kMinGraspHeight = 0.254;

// Success thresholds of a cycle: while the fold is held, the held corner lies
// within this horizontal distance of the lower corner, and the lower corner
// has slid less than this since the fold began.
constexpr double kFoldedCornerGap = 0.08;
constexpr double kMaxLowerCornerSlip = 0.02;

using ArmPose = std::array<double, 6>;

// An elbow-up arm pose near the start, from which the waypoints are solved.
constexpr ArmPose kIkSeed{-0.386, 2.708, -2.475, 0.0, 1.100, 1.185};

enum PhaseIndex : std::size_t { kHold, kLay, kFold, kHoldFold, kUnfold, kLift, kSettle, kPhaseCount };

struct PhaseInfo {
  const char* name;
  double duration;
};

inline constexpr std::array<PhaseInfo, kPhaseCount> kPhases{{
    {"Hold", 1.0},
    {"Lay down", 3.0},
    {"Fold over", 3.0},
    {"Hold the fold", 1.0},
    {"Unfold", 3.0},
    {"Lift", 3.0},
    {"Settle", 1.0},
}};

inline constexpr double kCycleDuration = [] {
  double duration = 0.0;
  for (const auto& phase : kPhases) duration += phase.duration;
  return duration;
}();

// Cloth and contact parameters. The defaults are the example's. The cloth
// slides easily over itself, so the folded half settles onto the lower half
// without dragging it. The cloth is solved together with the arm, the pads and
// the table (World::setMonolithicDeformables()), in ten sweeps per step.
struct Config {
  bool monolithic = true;
  double pad_friction = 0.8;
  double table_friction = 0.6;
  double cloth_friction = 0.3;
  double bend_stiffness = 5.0e-5;
  int iterations = 10;
  int substeps = 1;
};

struct CycleResult {
  bool held = true;     // both pads touched the cloth at every step
  bool laid = false;    // the lower corner rested on the table when the fold began
  bool folded = false;  // the held corner lay over the lower corner, which had not slid
  bool lifted = false;  // the whole cloth hung above the table again at the end
  double corner_gap = 0.0;
  double lower_corner_slip = 0.0;
  double max_mimic_error = 0.0;

  bool IsSuccessful() const { return held && laid && folded && lifted; }
};

// Joint position and velocity on a path at the time-scaling value s (0 to 1)
// changing at rate ds/dt: a cubic Hermite spline through the waypoints, with
// central-difference tangents inside and one-sided ones at the ends (two
// waypoints give a straight line). The arm passes every waypoint without
// stopping.
inline void SamplePath(const std::vector<ArmPose>& path, double s, double rate, std::size_t joint, double& position, double& velocity) {
  const std::size_t segments = path.size() - 1;
  if (segments == 0) {
    position = path.front()[joint];
    velocity = 0.0;
    return;
  }
  const double u = s * static_cast<double>(segments);
  const std::size_t i = std::min(static_cast<std::size_t>(u), segments - 1);
  const double t = u - static_cast<double>(i);
  auto tangent = [&](std::size_t k) {
    const std::size_t lo = k == 0 ? 0 : k - 1;
    const std::size_t hi = std::min(k + 1, segments);
    return (path[hi][joint] - path[lo][joint]) / static_cast<double>(hi - lo);
  };
  const double p0 = path[i][joint], p1 = path[i + 1][joint], m0 = tangent(i), m1 = tangent(i + 1);
  const double t2 = t * t, t3 = t2 * t;
  position = (2.0 * t3 - 3.0 * t2 + 1.0) * p0 + (t3 - 2.0 * t2 + t) * m0 + (-2.0 * t3 + 3.0 * t2) * p1 + (t3 - t2) * m1;
  const double slope = (6.0 * t2 - 6.0 * t) * p0 + (3.0 * t2 - 4.0 * t + 1.0) * m0 + (-6.0 * t2 + 6.0 * t) * p1 + (3.0 * t2 - 2.0 * t) * m1;
  velocity = slope * static_cast<double>(segments) * rate;
}

class Scene {
 public:
  explicit Scene(const std::string& model, const Config& config = Config());

  // Invoke under RaisimServer's world mutex immediately before integration.
  // Simulation time drives the script, so viewer pause and single-step work.
  void Update();
  void Step() {
    Update();
    world_.integrate();
  }

  raisim::World& world() { return world_; }
  raisim::ArticulatedSystem* robot() const { return robot_; }
  raisim::DeformableObject* cloth() const { return cloth_; }
  std::size_t phase_index() const { return phase_index_; }
  const CycleResult& last_result() const { return last_result_; }
  std::size_t completed_cycles() const { return completed_cycles_; }
  std::size_t held_corner() const { return held_corner_; }
  std::size_t lower_corner() const { return lower_corner_; }

 private:
  static constexpr std::array<double, 5> kFollowerSigns{-1.0, 1.0, -1.0, -1.0, 1.0};

  ArmPose SolveIk(const Eigen::Vector3d& position, const ArmPose& seed);
  void PlanFold();
  void AddCloth(const Config& config);
  void SetGrip(double grip);
  void Observe();

  raisim::World world_;
  // a second copy of the arm, used only to solve inverse kinematics
  raisim::World kinematics_;
  raisim::ArticulatedSystem* robot_ = nullptr;
  raisim::ArticulatedSystem* kinematic_arm_ = nullptr;
  raisim::DeformableObject* cloth_ = nullptr;
  std::size_t held_corner_ = 0;
  std::size_t lower_corner_ = 0;
  std::size_t phase_index_ = 0;
  std::array<std::vector<ArmPose>, kPhaseCount> paths_;
  raisim::Vec<3> touchdown_;
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

inline Scene::Scene(const std::string& model, const Config& config) {
  world_.setMonolithicDeformables(config.monolithic);
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

  raisim::ArticulatedSystemOption options;
  options.convexifyCollisionMeshes = true;
  robot_ = world_.addArticulatedSystem(model, "", {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);
  robot_->setName("kinova_robotiq");
  kinematic_arm_ = kinematics_.addArticulatedSystem(model);
  if (robot_->getDOF() != 12 || robot_->getGeneralizedCoordinateDim() != 12) {
    throw std::runtime_error("The fixed-base Kinova/Robotiq model must have twelve joints");
  }
  for (std::size_t i = 0; i < arm_indices_.size(); ++i) {
    arm_indices_[i] = robot_->getGeneralizedVelocityIndex("kinova_joint_" + std::to_string(i + 1));
  }
  grip_index_ = robot_->getGeneralizedVelocityIndex("robotiq_85_left_knuckle_joint");
  left_pad_ = robot_->getBodyIdx("robotiq_85_left_finger_tip_link");
  right_pad_ = robot_->getBodyIdx("robotiq_85_right_finger_tip_link");
  for (auto& body : robot_->getCollisionBodies()) {
    body.setMaterial(body.localIdx == left_pad_ || body.localIdx == right_pad_ ? "pad" : "arm");
  }
  const std::array<const char*, 5> followers{"robotiq_85_right_knuckle_joint", "robotiq_85_left_inner_knuckle_joint", "robotiq_85_right_inner_knuckle_joint", "robotiq_85_left_finger_tip_joint",
                                             "robotiq_85_right_finger_tip_joint"};
  for (std::size_t i = 0; i < followers.size(); ++i) {
    follower_indices_[i] = robot_->getGeneralizedVelocityIndex(followers[i]);
  }

  // The fold arc depends on where the cloth lands, so PlanFold() solves it
  // when the fold begins. Unfolding and lifting retrace the same paths.
  const ArmPose start = SolveIk(kStartPosition, kIkSeed);
  const ArmPose lay_mid = SolveIk(kLayMidPosition, start);
  const ArmPose fold_top = SolveIk(kFoldTopPosition, lay_mid);
  paths_[kHold] = {start};
  paths_[kLay] = {start, lay_mid, fold_top};
  paths_[kLift] = {fold_top, lay_mid, start};
  paths_[kSettle] = {start};

  target_ = Eigen::VectorXd::Zero(robot_->getDOF());
  velocity_target_ = target_;
  Eigen::VectorXd p_gain = target_;
  Eigen::VectorXd d_gain = target_;
  const ArmPose p{2000.0, 3000.0, 2500.0, 600.0, 600.0, 300.0};
  const ArmPose d{60.0, 80.0, 70.0, 20.0, 20.0, 12.0};
  for (std::size_t i = 0; i < arm_indices_.size(); ++i) {
    p_gain[arm_indices_[i]] = p[i];
    d_gain[arm_indices_[i]] = d[i];
    target_[arm_indices_[i]] = start[i];
  }
  p_gain[grip_index_] = 10.0;
  d_gain[grip_index_] = 0.2;
  SetGrip(kInitialGrip);
  robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  robot_->setGeneralizedCoordinate(target_);
  robot_->setGeneralizedVelocity(velocity_target_);
  robot_->setGeneralizedForce(Eigen::VectorXd::Zero(robot_->getDOF()));
  robot_->setPdGains(p_gain, d_gain);
  SetGrip(kClosed);
  robot_->setPdTarget(target_, velocity_target_);

  AddCloth(config);
  world_.setMaterialPairProp("pad", "cloth", config.pad_friction, 0.0, 0.001);
  world_.setMaterialPairProp("table", "cloth", config.table_friction, 0.0, 0.001);
  world_.setMaterialPairProp("arm", "cloth", 0.3, 0.0, 0.001);
  world_.setMaterialPairProp("cloth", "cloth", config.cloth_friction, 0.0, 0.001);

  // Let the pads squeeze the corner and the cloth come to rest; repeat the
  // complete physical initial state, including solver history, at the end of
  // each cycle.
  for (int i = 0; i < 1000; ++i) world_.integrate();
  checkpoint_ = world_.captureCheckpoint();
  start_time_ = world_.getWorldTime();
}

// grasp_frame at `position` on the kinematic copy, pointing down with its x
// axis (the closing direction) along world y.
inline ArmPose Scene::SolveIk(const Eigen::Vector3d& position, const ArmPose& seed) {
  raisim::Mat<3, 3> orientation;
  orientation.e() << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0;
  raisim::ArticulatedSystem::IKOptions options;
  options.maxIterations = 200;
  options.positionTolerance = 1e-8;
  options.orientationTolerance = 1e-8;
  options.seed.assign(kinematic_arm_->getGeneralizedCoordinateDim(), 0.0);
  for (std::size_t i = 0; i < arm_indices_.size(); ++i) options.seed[arm_indices_[i]] = seed[i];
  const raisim::Vec<3> target{position[0], position[1], position[2]};
  const auto result = kinematic_arm_->solveIK("grasp_frame_joint", target, orientation, options);
  if (!result.success) {
    throw std::runtime_error("The arm cannot reach a cloth-folding waypoint: " + result.message);
  }
  ArmPose pose;
  for (std::size_t i = 0; i < arm_indices_.size(); ++i) pose[i] = result.solution[arm_indices_[i]];
  return pose;
}

// The lower half lies on the table; its fold line is half a diagonal from the
// corner that touched down first. The fold arc starts where the lay-down
// ended and circles that line.
inline void Scene::PlanFold() {
  touchdown_ = cloth_->getPositions()[lower_corner_];
  current_.laid = touchdown_[2] < kTableHeight + 1.5 * kClothRadius;
  const double fold_line = touchdown_[1] - kHalfDiagonal;
  std::vector<ArmPose> fold{paths_[kLay].back()};
  for (double angle : kFoldAngles) {
    const Eigen::Vector3d position{kStartPosition[0], fold_line + kFoldRadius * std::sin(angle),
                                   std::max(kMinGraspHeight, kTableHeight + kClothRadius + kFoldRadius * std::cos(angle) + kPadEdgeDepth)};
    fold.push_back(SolveIk(position, fold.back()));
  }
  paths_[kFold] = fold;
  paths_[kHoldFold] = {fold.back()};
  paths_[kUnfold].assign(fold.rbegin(), fold.rend());
}

inline void Scene::AddCloth(const Config& config) {
  // the held corner sits in the upper half of the pads, above grasp_frame
  raisim::Vec<3> grasp;
  robot_->getFramePosition("grasp_frame_joint", grasp);
  const raisim::Vec<3> top{grasp[0], grasp[1], grasp[2] + 0.008};
  const int n = kClothResolution;
  const double spacing = kClothSide / (n - 1);
  const double c = std::sqrt(0.5);
  std::vector<raisim::Vec<3>> vertices;
  std::vector<raisim::DeformableObject::Triangle> triangles;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const double u = spacing * i, v = spacing * j;
      vertices.push_back(raisim::Vec<3>{top[0] + c * (v - u), top[1], top[2] - c * (u + v)});
    }
  }
  for (int j = 0; j + 1 < n; ++j) {
    for (int i = 0; i + 1 < n; ++i) {
      const auto a = static_cast<unsigned>(j * n + i);
      const unsigned b = a + 1, cc = a + n, dd = cc + 1;
      triangles.push_back({a, b, dd});
      triangles.push_back({a, dd, cc});
    }
  }
  held_corner_ = 0;
  lower_corner_ = vertices.size() - 1;

  // A light cotton square: nearly inextensible edges, a soft bending modulus,
  // and self-collision so that the folded half rests on the lower half.
  raisim::DeformableObject::Material material;
  material.totalMass = 0.02;
  material.distanceCompliance = 1.0e-6;
  material.bendStiffness = config.bend_stiffness;
  material.damping = 0.002;
  material.airDrag = 0.5;
  material.collisionRadius = kClothRadius;
  material.iterations = config.iterations;
  material.substeps = config.substeps;
  material.selfCollision = true;
  cloth_ = world_.addDeformableCloth(vertices, triangles, {}, material, "cloth");
  cloth_->setName("cloth");
}

inline void Scene::SetGrip(double grip) {
  target_[grip_index_] = grip;
  velocity_target_[grip_index_] = 0.0;
  // Mimic followers have zero PD gains; their targets document the same
  // linkage relation and never provide extra squeeze torques.
  for (std::size_t j = 0; j < follower_indices_.size(); ++j) {
    target_[follower_indices_[j]] = kFollowerSigns[j] * grip;
    velocity_target_[follower_indices_[j]] = 0.0;
  }
}

inline void Scene::Update() {
  Observe();
  double time = world_.getWorldTime() - start_time_;
  if (time >= kCycleDuration - 0.5 * kTimeStep) {
    last_result_ = current_;
    ++completed_cycles_;
    std::string error;
    if (!world_.restoreCheckpoint(checkpoint_, &error)) {
      throw std::runtime_error("Cannot reset the cloth-folding world: " + error);
    }
    start_time_ = world_.getWorldTime();
    current_ = CycleResult{};
    time = 0.0;
  }
  double begin = 0.0;
  for (std::size_t i = 0; i < kPhases.size(); ++i) {
    if (time < begin + kPhases[i].duration || i + 1 == kPhases.size()) {
      if (i == kFold && phase_index_ != kFold) PlanFold();
      phase_index_ = i;
      const double s = std::clamp((time - begin) / kPhases[i].duration, 0.0, 1.0);
      const double blend = s * s * s * (10.0 + s * (-15.0 + 6.0 * s));
      const double rate = 30.0 * s * s * (1.0 - s) * (1.0 - s) / kPhases[i].duration;
      for (std::size_t j = 0; j < arm_indices_.size(); ++j) {
        SamplePath(paths_[i], blend, rate, j, target_[arm_indices_[j]], velocity_target_[arm_indices_[j]]);
      }
      robot_->setPdTarget(target_, velocity_target_);
      break;
    }
    begin += kPhases[i].duration;
  }
}

inline void Scene::Observe() {
  const auto& q = robot_->getGeneralizedCoordinate();
  if (!q.e().allFinite() || !robot_->getGeneralizedVelocity().e().allFinite()) {
    throw std::runtime_error("Non-finite robot state");
  }
  double lowest = 1.0;
  for (const auto& position : cloth_->getPositions()) {
    if (!position.e().allFinite()) throw std::runtime_error("Non-finite cloth state");
    lowest = std::min(lowest, position[2] - kTableHeight);
  }
  for (std::size_t j = 0; j < follower_indices_.size(); ++j) {
    current_.max_mimic_error = std::max(current_.max_mimic_error, std::abs(q[follower_indices_[j]] - kFollowerSigns[j] * q[grip_index_]));
  }
  bool left = false;
  bool right = false;
  for (const auto& contact : robot_->getContacts()) {
    if (contact.skip() || contact.getPairObjectIndex() != cloth_->getIndexInWorld()) continue;
    left |= contact.getlocalBodyIndex() == left_pad_;
    right |= contact.getlocalBodyIndex() == right_pad_;
  }
  current_.held &= left && right;
  if (phase_index_ == kHoldFold) {
    const auto& held = cloth_->getPositions()[held_corner_];
    const auto& lower = cloth_->getPositions()[lower_corner_];
    current_.corner_gap = std::hypot(held[0] - lower[0], held[1] - lower[1]);
    current_.lower_corner_slip = std::hypot(lower[0] - touchdown_[0], lower[1] - touchdown_[1]);
    current_.folded = current_.corner_gap < kFoldedCornerGap && held[2] > lower[2] + kClothRadius && current_.lower_corner_slip < kMaxLowerCornerSlip;
  }
  if (phase_index_ == kSettle) current_.lifted = lowest > kClothRadius;
}

}  // namespace raisim_examples::cloth_fold

#endif  // RAISIM_EXAMPLES_INCLUDE_ROBOTIQ_CLOTH_FOLD_HPP_
