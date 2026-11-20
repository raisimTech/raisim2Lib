// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#ifndef RAISIM_EXAMPLES_INCLUDE_MAGNETIC_WALL_CLIMBER_HPP_
#define RAISIM_EXAMPLES_INCLUDE_MAGNETIC_WALL_CLIMBER_HPP_

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

// A Go1 with magnetic feet trots up and down a steel wall, after KAIST's MARVEL
// (Hong et al., "Agile and versatile climbing on ferromagnetic surfaces with a
// quadrupedal robot", Science Robotics, 2022). Each foot carries a switchable
// magnet. The controller switches a magnet off just before its foot lifts and
// on again as the foot lands, so the diagonal pair of feet on the wall always
// holds the robot.
namespace raisim_examples::wall_climber {

constexpr double kTimeStep = 0.001;
constexpr double kPi = 3.14159265358979323846;

// MARVEL's foot: the magnet holds a foot to steel with 530 N, and the sole, a
// rubber filled with iron particles, resists 440 N of sliding.
constexpr double kMagnetPull = 530.0;
constexpr double kMagnetRange = 0.01;
constexpr double kSoleFriction = 440.0 / 530.0;

// Go1 leg geometry (go1.urdf).
constexpr double kHipX = 0.1881;
constexpr double kHipY = 0.04675;
constexpr double kHipOffset = 0.08;
constexpr double kThigh = 0.213;
constexpr double kCalf = 0.213;
constexpr double kFootRadius = 0.02;

// The wall's face is the plane x = 0; the robot climbs along +z.
constexpr double kWallHeight = 4.0;
constexpr double kWallWidth = 3.0;
// Distance between the base and the wall [m].
constexpr double kBaseHeight = 0.22;
// The base climbs between these heights above the ground [m].
constexpr double kBottom = 0.75;
constexpr double kTop = 3.1;
constexpr double kClimbSpeed = 0.5;
constexpr double kClimbAcceleration = 0.8;
constexpr double kHoldTime = 1.5;

// Trot: diagonal legs move together, each on the wall for kDuty of the period.
// Both pairs stand together for a short time, during which a magnet lets go.
constexpr double kPeriod = 0.3;
constexpr double kDuty = 0.7;
constexpr double kClearance = 0.04;
// A magnet switches off this long before its foot lifts [s].
constexpr double kRelease = 0.02;
// A landing foot switches its magnet on for the last part of its swing.
constexpr double kCatch = 0.75;
// Feet are placed this far into the wall, so the legs press them on.
constexpr double kPreload = 0.003;

constexpr double kJointP = 120.0;
constexpr double kJointD = 3.0;

constexpr std::array<const char*, 4> kLegNames{"FR", "FL", "RR", "RL"};

/// Joint angles (abduction, hip, knee) that put a Go1 foot center at `foot`,
/// given in the base frame relative to the leg's abduction joint. `side` is
/// +1 for a left leg and -1 for a right leg.
inline Eigen::Vector3d legIk(const Eigen::Vector3d& foot, double side) {
  const double offset = side * kHipOffset;
  const double radial = foot.y() * foot.y() + foot.z() * foot.z();
  const double length = std::sqrt(std::max(radial - offset * offset, 1e-8));
  const double abduction = std::atan2(foot.z(), foot.y()) + std::atan2(length, offset);
  const double reach = foot.x() * foot.x() + length * length;
  const double cosKnee = std::clamp((reach - kThigh * kThigh - kCalf * kCalf) / (2.0 * kThigh * kCalf), -1.0, 1.0);
  const double knee = -std::acos(cosKnee);
  const double hip = std::atan2(-foot.x(), length) - std::atan2(kCalf * std::sin(knee), kThigh + kCalf * std::cos(knee));
  return {abduction, hip, knee};
}

/// The robot on its wall and the climbing controller.
class Scene {
 public:
  explicit Scene(const std::string& go1Urdf, bool magnets = true);

  /// Runs the controller and integrates one step.
  void step();

  raisim::World& world() { return world_; }
  raisim::ArticulatedSystem* robot() const { return robot_; }
  /// Height of the base above the ground, along the wall [m].
  double height() const { return robot_->getBasePosition()[2]; }
  /// Height the controller moves the base to [m].
  double targetHeight() const { return bodyHeight_; }
  /// Gripper index of the magnet of a leg (kLegNames order).
  std::size_t magnet(std::size_t leg) const { return legs_[leg].magnet; }
  /// Whether the magnet of a leg is switched on.
  bool magnetOn(std::size_t leg) const { return robot_->isGripperActive(legs_[leg].magnet); }
  /// Whether the magnet of a leg pulled on the wall in the last step.
  bool magnetHolds(std::size_t leg) const { return robot_->getGripperState(legs_[leg].magnet).engaged; }
  /// Whether a leg's foot is on the wall in the gait (not swinging).
  bool inStance(std::size_t leg) const { return legs_[leg].stance; }
  /// Center of a foot in the world frame [m].
  Eigen::Vector3d footPosition(std::size_t leg) const;
  /// Where a foot was placed on the wall, while it is in stance [m].
  const Eigen::Vector3d& foothold(std::size_t leg) const { return legs_[leg].foothold; }
  /// Where the controller puts the center of a foot [m].
  const Eigen::Vector3d& footTarget(std::size_t leg) const { return legs_[leg].target; }
  /// The number of times the robot has reached the top of its climb.
  int climbs() const { return climbs_; }

 private:
  struct Leg {
    Eigen::Vector3d hip;
    double side = 1.0;
    double phaseOffset = 0.0;
    std::size_t joint = 0;
    std::size_t foot = 0;
    std::size_t magnet = 0;
    bool stance = true;
    Eigen::Vector3d foothold = Eigen::Vector3d::Zero();
    Eigen::Vector3d liftoff = Eigen::Vector3d::Zero();
    Eigen::Vector3d target = Eigen::Vector3d::Zero();
  };

  void updateClimb();
  void advanceGait();
  void control();
  Eigen::Vector3d nominalFoot(const Leg& leg) const;
  Eigen::Vector3d bodyPosition() const { return {kBaseHeight, 0.0, bodyHeight_}; }

  raisim::World world_;
  raisim::ArticulatedSystem* robot_ = nullptr;
  std::array<Leg, 4> legs_;
  Eigen::Matrix3d wallOrientation_;
  bool magnets_ = true;
  double mass_ = 0.0;

  double bodyHeight_ = kBottom;
  double speed_ = 0.0;
  double goal_ = kBottom;
  double holdTimer_ = 0.0;
  int climbs_ = 0;

  double phase_ = 0.05;
  bool walking_ = false;

  Eigen::VectorXd target_, targetVelocity_, feedforward_;
  Eigen::MatrixXd jacobian_;
};

inline Scene::Scene(const std::string& go1Urdf, bool magnets) : magnets_(magnets) {
  world_.setTimeStep(kTimeStep);
  world_.setMagneticMaterial("steel");
  world_.addGround(0.0, "concrete");
  auto* wall = world_.addBox(0.3, kWallWidth, kWallHeight, 1.0, "steel");
  wall->setName("steel_wall");
  wall->setBodyType(raisim::BodyType::STATIC);
  wall->setPosition(-0.15, 0.0, 0.5 * kWallHeight);
  wall->setAppearance("0.45,0.48,0.52,1");
  world_.setMaterialPairProp("steel", "magnet_sole", kSoleFriction, 0.0, 0.0);

  robot_ = world_.addArticulatedSystem(go1Urdf);
  robot_->setName("go1");
  if (robot_->getDOF() != 18) throw std::runtime_error("The climber needs the floating-base Go1 with twelve joints");
  mass_ = robot_->getTotalMass();

  // The base faces up the wall, its belly toward the steel.
  wallOrientation_.col(0) = Eigen::Vector3d::UnitZ();
  wallOrientation_.col(1) = -Eigen::Vector3d::UnitY();
  wallOrientation_.col(2) = Eigen::Vector3d::UnitX();

  // The sole faces the wall when the leg stands at its nominal pose, so it is
  // tilted against the shank by the shank's pitch there.
  const double side[4] = {-1.0, 1.0, -1.0, 1.0};
  const double front[4] = {1.0, 1.0, -1.0, -1.0};
  std::vector<raisim::GripperDefinition> magnetDefs;
  for (std::size_t i = 0; i < legs_.size(); ++i) {
    auto& leg = legs_[i];
    const std::string name = kLegNames[i];
    leg.hip = Eigen::Vector3d(front[i] * kHipX, side[i] * kHipY, 0.0);
    leg.side = side[i];
    leg.phaseOffset = (i == 0 || i == 3) ? 0.0 : 0.5;
    leg.joint = robot_->getGeneralizedVelocityIndex(name + "_hip_joint");
    leg.foot = robot_->getFrameIdxByName(name + "_foot_fixed");
    const Eigen::Vector3d joints = legIk(nominalFoot(leg) - leg.hip, leg.side);
    const double shankPitch = joints[1] + joints[2];
    raisim::GripperDefinition magnet;
    magnet.name = name + "_magnet";
    magnet.link = name + "_foot";
    magnet.type = raisim::GripperDefinition::Type::MAGNETIC;
    magnet.axis = {std::sin(shankPitch), 0.0, -std::cos(shankPitch)};
    magnet.position = {kFootRadius * magnet.axis[0], 0.0, kFootRadius * magnet.axis[2]};
    magnet.maxForce = kMagnetPull;
    magnet.range = kMagnetRange;
    magnetDefs.push_back(magnet);
  }
  robot_->setGrippers(magnetDefs);
  for (std::size_t i = 0; i < legs_.size(); ++i) legs_[i].magnet = robot_->getGripperIndex(magnetDefs[i].name);
  for (auto& body : robot_->getCollisionBodies()) {
    if (body.getCollisionBody()->name.find("_foot") != std::string::npos) body.setMaterial("magnet_sole");
  }

  Eigen::VectorXd gc = Eigen::VectorXd::Zero(robot_->getGeneralizedCoordinateDim());
  gc.head<3>() = bodyPosition();
  const Eigen::Quaterniond orientation(wallOrientation_);
  gc.segment<4>(3) << orientation.w(), orientation.x(), orientation.y(), orientation.z();
  for (auto& leg : legs_) {
    const Eigen::Vector3d foot = nominalFoot(leg);
    gc.segment<3>(leg.joint + 1) = legIk(foot - leg.hip, leg.side);
    leg.foothold = bodyPosition() + wallOrientation_ * foot;
    leg.foothold.x() = kFootRadius - kPreload;
  }
  robot_->setGeneralizedCoordinate(gc);
  robot_->setGeneralizedVelocity(Eigen::VectorXd::Zero(robot_->getDOF()));

  Eigen::VectorXd pGain = Eigen::VectorXd::Constant(robot_->getDOF(), kJointP);
  Eigen::VectorXd dGain = Eigen::VectorXd::Constant(robot_->getDOF(), kJointD);
  pGain.head<6>().setZero();
  dGain.head<6>().setZero();
  robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  robot_->setPdGains(pGain, dGain);
  target_ = gc;
  targetVelocity_ = Eigen::VectorXd::Zero(robot_->getDOF());
  feedforward_ = Eigen::VectorXd::Zero(robot_->getDOF());
  jacobian_ = Eigen::MatrixXd::Zero(3, robot_->getDOF());
  robot_->setPdTarget(target_, targetVelocity_);
  for (std::size_t i = 0; i < legs_.size(); ++i) robot_->setGripperActive(legs_[i].magnet, magnets_);
}

inline Eigen::Vector3d Scene::nominalFoot(const Leg& leg) const {
  return leg.hip + Eigen::Vector3d(0.0, leg.side * kHipOffset, -(kBaseHeight - kFootRadius));
}

inline Eigen::Vector3d Scene::footPosition(std::size_t leg) const {
  raisim::Vec<3> position;
  robot_->getFramePosition(legs_[leg].foot, position);
  return position.e();
}

inline void Scene::step() {
  updateClimb();
  advanceGait();
  control();
  world_.integrate();
}

inline void Scene::updateClimb() {
  // Climb to the goal with a trapezoidal speed profile, rest there, and turn back.
  const double remaining = goal_ - bodyHeight_;
  if (std::abs(remaining) < 1e-4 && std::abs(speed_) < 1e-9) {
    speed_ = 0.0;
    if (walking_) return;
    holdTimer_ += kTimeStep;
    if (holdTimer_ < kHoldTime) return;
    holdTimer_ = 0.0;
    if (goal_ == kTop) climbs_++;
    goal_ = goal_ == kTop ? kBottom : kTop;
    return;
  }
  const double direction = remaining > 0.0 ? 1.0 : -1.0;
  const double brake = std::sqrt(2.0 * kClimbAcceleration * std::abs(remaining));
  const double desired = direction * std::min(kClimbSpeed, brake);
  speed_ += std::clamp(desired - speed_, -kClimbAcceleration * kTimeStep, kClimbAcceleration * kTimeStep);
  if (std::abs(speed_ * kTimeStep) >= std::abs(remaining)) {
    bodyHeight_ = goal_;
    speed_ = 0.0;
  } else {
    bodyHeight_ += speed_ * kTimeStep;
  }
}

inline void Scene::advanceGait() {
  // The gait runs while the base moves. Once it stops, it runs on until both
  // pairs stand on the wall, and stays there.
  const bool moving = std::abs(goal_ - bodyHeight_) > 1e-4 || std::abs(speed_) > 1e-9;
  if (moving) walking_ = true;
  if (!walking_) return;
  const double previous = phase_;
  phase_ = std::fmod(phase_ + kTimeStep / kPeriod, 1.0);
  if (!moving) {
    for (const double rest : {0.5 * (kDuty - 0.5), 0.5 + 0.5 * (kDuty - 0.5)}) {
      const bool crossed = previous < rest ? (phase_ >= rest || phase_ < previous) : (phase_ >= rest && phase_ < previous);
      if (crossed) {
        phase_ = rest;
        walking_ = false;
      }
    }
  }
}

inline void Scene::control() {
  const Eigen::Vector3d body = bodyPosition();
  const Eigen::Vector3d velocity(0.0, 0.0, speed_);
  const double swingTime = (1.0 - kDuty) * kPeriod;
  const double stanceTime = kDuty * kPeriod;

  // Switch the magnets first: a foot carries weight only while its magnet stays on.
  int holding = 0;
  std::array<bool, 4> holds{};
  std::array<double, 4> phases{};
  for (std::size_t i = 0; i < legs_.size(); ++i) {
    auto& leg = legs_[i];
    phases[i] = std::fmod(phase_ + leg.phaseOffset, 1.0);
    const bool stance = phases[i] < kDuty;
    if (stance && !leg.stance) {
      leg.foothold = footPosition(i);
      leg.foothold.x() = kFootRadius - kPreload;
    } else if (!stance && leg.stance) {
      leg.liftoff = footPosition(i);
    }
    leg.stance = stance;
    const bool magnet = stance ? !walking_ || phases[i] < kDuty - kRelease / kPeriod
                               : (phases[i] - kDuty) / (1.0 - kDuty) > kCatch;
    robot_->setGripperActive(leg.magnet, magnets_ && magnet);
    holds[i] = stance && magnetHolds(i);
    holding += holds[i] ? 1 : 0;
  }

  // Stance feet aim from the base pose the controller wants, so they move the
  // base there; swinging feet aim from the actual pose, so they land on the wall.
  const Eigen::Vector3d base = robot_->getBasePosition().e();
  const Eigen::Matrix3d baseOrientation = robot_->getBaseOrientation().e();
  const Eigen::VectorXd previousTarget = target_;
  feedforward_.setZero();
  for (std::size_t i = 0; i < legs_.size(); ++i) {
    auto& leg = legs_[i];
    Eigen::Vector3d foot = leg.foothold;
    if (!leg.stance) {
      const double progress = (phases[i] - kDuty) / (1.0 - kDuty);
      const double remaining = (1.0 - progress) * swingTime;
      Eigen::Vector3d landing = body + velocity * (remaining + 0.5 * stanceTime) + wallOrientation_ * nominalFoot(leg);
      landing.x() = kFootRadius - kPreload;
      const double blend = progress * progress * progress * (10.0 - 15.0 * progress + 6.0 * progress * progress);
      foot = leg.liftoff + blend * (landing - leg.liftoff);
      foot.x() += kClearance * std::sin(kPi * progress);
    }
    leg.target = foot;
    const Eigen::Vector3d inBase = leg.stance ? Eigen::Vector3d(wallOrientation_.transpose() * (foot - body))
                                              : Eigen::Vector3d(baseOrientation.transpose() * (foot - base));
    target_.segment<3>(leg.joint + 1) = legIk(inBase - leg.hip, leg.side);

    // The feet that hold carry the weight: each pushes the wall down with its share.
    if (holds[i]) {
      robot_->getDenseFrameJacobian(leg.foot, jacobian_);
      const Eigen::Vector3d push = mass_ * world_.getGravity().e() / holding;
      feedforward_.segment<3>(leg.joint) = jacobian_.block(0, leg.joint, 3, 3).transpose() * push;
    }
  }
  targetVelocity_.tail(12) = ((target_ - previousTarget) / kTimeStep).tail(12).cwiseMax(-20.0).cwiseMin(20.0);
  robot_->setPdTarget(target_, targetVelocity_);
  robot_->setGeneralizedForce(feedforward_);
}

}  // namespace raisim_examples::wall_climber

#endif  // RAISIM_EXAMPLES_INCLUDE_MAGNETIC_WALL_CLIMBER_HPP_
