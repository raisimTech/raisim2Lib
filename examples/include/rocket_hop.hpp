// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#ifndef RAISIM_EXAMPLES_INCLUDE_ROCKET_HOP_HPP_
#define RAISIM_EXAMPLES_INCLUDE_ROCKET_HOP_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "raisim/World.hpp"

// A rocket hops between two landing pads: it lifts off, climbs, flies sideways,
// descends and lands on its legs, then flies back. Its main engine hangs from a
// two-axis gimbal; tilting the engine steers the rocket, as on real launchers.
// Four small thrusters at the top hold the roll against the swirl of the
// engine's exhaust.
namespace raisim_examples::rocket_hop {

constexpr double kTimeStep = 0.001;
constexpr double kGravity = 9.81;

// rocket.urdf: the engine and its limits.
constexpr double kMaxThrust = 12000.0;
// The engine cannot throttle below this; it is either off or above it [N].
constexpr double kMinThrust = 0.4 * kMaxThrust;
constexpr double kMaxGimbal = 0.14;
// The nozzle's exit is this far below the gimbal [m].
constexpr double kNozzleLength = 0.5;
constexpr double kMaxRollThrust = 60.0;
constexpr double kRollArm = 0.36;

// Two landing pads, their tops at kPadTop, and the flight between them.
constexpr double kPadRadius = 2.5;
constexpr double kPadTop = 0.2;
constexpr std::array<double, 2> kPadA{0.0, 0.0};
constexpr std::array<double, 2> kPadB{20.0, 0.0};
constexpr double kCruiseHeight = 25.0;
// The rocket stands with its foot pads this far below its base frame [m].
constexpr double kFootDepth = 0.56;
constexpr double kMaxTilt = 0.26;
// The last metre down to the pad is flown at this sinking speed [m/s].
constexpr double kSinkSpeed = 0.35;
constexpr double kSinkHeight = 1.0;

/// A trajectory piece with zero velocity and acceleration at both ends.
struct Segment {
  Eigen::Vector3d from = Eigen::Vector3d::Zero();
  Eigen::Vector3d to = Eigen::Vector3d::Zero();
  double duration = 1.0;

  void evaluate(double time, Eigen::Vector3d& position, Eigen::Vector3d& velocity, Eigen::Vector3d& acceleration) const {
    const double s = std::clamp(time / duration, 0.0, 1.0);
    const double shape = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s);
    const double rate = 30.0 * s * s * (1.0 - s) * (1.0 - s) / duration;
    const double curvature = 60.0 * s * (1.0 - s) * (1.0 - 2.0 * s) / (duration * duration);
    const Eigen::Vector3d delta = to - from;
    position = from + shape * delta;
    velocity = rate * delta;
    acceleration = curvature * delta;
  }
};

/// What the rocket is doing.
enum class Phase { kOnPad, kAscent, kCruise, kDescent, kTouchdown };

/// The rocket, its pads and the flight controller.
class Scene {
 public:
  explicit Scene(const std::string& rocketUrdf);

  /// Runs the guidance and control and integrates one step.
  void step();

  raisim::World& world() { return world_; }
  raisim::ArticulatedSystem* rocket() const { return rocket_; }
  Phase phase() const { return phase_; }
  bool engineOn() const { return engineOn_; }
  /// The thrust of the main engine [N].
  double thrust() const { return rocket_->getThrust(engine_); }
  std::size_t engine() const { return engine_; }
  /// Thruster indices of the roll thrusters.
  const std::array<std::size_t, 4>& rollThrusters() const { return roll_; }
  /// Center of mass the controller follows, and where it is [m].
  const Eigen::Vector3d& reference() const { return referencePosition_; }
  Eigen::Vector3d centerOfMass() const { return rocket_->getCOM().e(); }
  /// Angle between the rocket's axis and the vertical [rad].
  double tilt() const { return std::acos(std::clamp(rocket_->getBaseOrientation()(2, 2), -1.0, 1.0)); }
  /// Angle between the fins and the world's x-z and y-z planes, about the rocket's axis [rad].
  double rollError() const;
  /// The number of landings so far.
  int landings() const { return landings_; }
  /// Sinking speed of the center of mass just before the legs last touched the pad [m/s].
  double touchdownSpeed() const { return touchdownSpeed_; }

 private:
  void beginFlight();
  void guide(Eigen::Vector3d& position, Eigen::Vector3d& velocity, Eigen::Vector3d& acceleration);
  void control();
  Eigen::Vector3d restingCenterOfMass(const std::array<double, 2>& pad) const;
  bool onGround() const;

  raisim::World world_;
  raisim::ArticulatedSystem* rocket_ = nullptr;
  std::size_t engine_ = 0;
  std::array<std::size_t, 4> roll_{};
  std::size_t gimbalX_ = 0, gimbalY_ = 0;
  double comHeight_ = 0.0;

  Phase phase_ = Phase::kOnPad;
  Segment segment_;
  double phaseTime_ = 0.0;
  bool engineOn_ = false;
  bool toB_ = true;
  int landings_ = 0;
  double touchdownSpeed_ = 0.0;
  double sinkSpeed_ = 0.0;
  Eigen::Vector3d referencePosition_ = Eigen::Vector3d::Zero();
  Eigen::VectorXd gimbalTarget_, gimbalVelocity_;
};

inline Scene::Scene(const std::string& rocketUrdf) {
  world_.setTimeStep(kTimeStep);
  world_.addGround(0.0, "concrete");
  const std::array<std::array<double, 2>, 2> pads{kPadA, kPadB};
  for (std::size_t i = 0; i < pads.size(); ++i) {
    auto* slab = world_.addCylinder(kPadRadius, kPadTop, 1.0, "concrete");
    slab->setName(i == 0 ? "pad_a" : "pad_b");
    slab->setBodyType(raisim::BodyType::STATIC);
    slab->setPosition(pads[i][0], pads[i][1], 0.5 * kPadTop);
    slab->setAppearance("0.36,0.37,0.38,1");
  }

  rocket_ = world_.addArticulatedSystem(rocketUrdf);
  rocket_->setName("rocket");
  engine_ = rocket_->getThrusterIndex("main_engine");
  const std::array<const char*, 4> rollNames{"rcs_roll_left_0", "rcs_roll_left_1", "rcs_roll_right_0", "rcs_roll_right_1"};
  for (std::size_t i = 0; i < roll_.size(); ++i) roll_[i] = rocket_->getThrusterIndex(rollNames[i]);
  if (engine_ == std::size_t(-1) || std::find(roll_.begin(), roll_.end(), std::size_t(-1)) != roll_.end())
    throw std::runtime_error("rocket.urdf must define the main engine and four roll thrusters");
  gimbalX_ = rocket_->getGeneralizedVelocityIndex("gimbal_x");
  gimbalY_ = rocket_->getGeneralizedVelocityIndex("gimbal_y");

  Eigen::VectorXd gc = Eigen::VectorXd::Zero(rocket_->getGeneralizedCoordinateDim());
  gc << kPadA[0], kPadA[1], kPadTop + kFootDepth, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  rocket_->setGeneralizedCoordinate(gc);
  rocket_->setGeneralizedVelocity(Eigen::VectorXd::Zero(rocket_->getDOF()));
  comHeight_ = rocket_->getCOM()[2] - gc[2];

  // The gimbal actuators are stiff servos; the base is free.
  Eigen::VectorXd pGain = Eigen::VectorXd::Zero(rocket_->getDOF());
  Eigen::VectorXd dGain = Eigen::VectorXd::Zero(rocket_->getDOF());
  pGain[gimbalX_] = pGain[gimbalY_] = 20000.0;
  dGain[gimbalX_] = dGain[gimbalY_] = 200.0;
  rocket_->setPdGains(pGain, dGain);
  gimbalTarget_ = gc;
  gimbalVelocity_ = Eigen::VectorXd::Zero(rocket_->getDOF());
  rocket_->setPdTarget(gimbalTarget_, gimbalVelocity_);
  referencePosition_ = restingCenterOfMass(kPadA);
}

inline Eigen::Vector3d Scene::restingCenterOfMass(const std::array<double, 2>& pad) const {
  return {pad[0], pad[1], kPadTop + kFootDepth + comHeight_};
}

inline double Scene::rollError() const {
  const Eigen::Matrix3d orientation = rocket_->getBaseOrientation().e();
  const Eigen::Vector3d axis = orientation.col(2);
  const Eigen::Vector3d heading = (Eigen::Vector3d::UnitX() - axis.x() * axis).normalized();
  const Eigen::Vector3d finAxis = orientation.col(0);
  return std::atan2(heading.cross(finAxis).dot(axis), finAxis.dot(heading));
}

inline bool Scene::onGround() const {
  for (const auto& contact : rocket_->getContacts())
    if (!contact.skip()) return true;
  return false;
}

inline void Scene::step() {
  phaseTime_ += kTimeStep;
  control();
  world_.integrate();
}

inline void Scene::beginFlight() {
  const auto& from = toB_ ? kPadA : kPadB;
  segment_ = {restingCenterOfMass(from), restingCenterOfMass(from) + Eigen::Vector3d(0, 0, kCruiseHeight), 9.0};
  phase_ = Phase::kAscent;
  phaseTime_ = 0.0;
  engineOn_ = true;
}

inline void Scene::guide(Eigen::Vector3d& position, Eigen::Vector3d& velocity, Eigen::Vector3d& acceleration) {
  // Each phase follows one smooth segment; the next starts where it ended.
  const auto& to = toB_ ? kPadB : kPadA;
  const Eigen::Vector3d landing = restingCenterOfMass(to);
  if (phase_ == Phase::kOnPad) {
    position = restingCenterOfMass(toB_ ? kPadA : kPadB);
    velocity.setZero();
    acceleration.setZero();
    if (phaseTime_ > 3.0) beginFlight();
    return;
  }
  if (phaseTime_ >= segment_.duration && phase_ != Phase::kTouchdown) {
    const Eigen::Vector3d end = segment_.to;
    phaseTime_ = 0.0;
    if (phase_ == Phase::kAscent) {
      segment_ = {end, landing + Eigen::Vector3d(0, 0, kCruiseHeight), 9.0};
      phase_ = Phase::kCruise;
    } else if (phase_ == Phase::kCruise) {
      segment_ = {end, landing + Eigen::Vector3d(0, 0, kSinkHeight), 9.0};
      phase_ = Phase::kDescent;
    } else {
      segment_.from = end;
      phase_ = Phase::kTouchdown;
    }
  }
  if (phase_ != Phase::kTouchdown) {
    segment_.evaluate(phaseTime_, position, velocity, acceleration);
    return;
  }
  // Sink at a steady speed, reached within kRamp, until the legs touch the pad.
  constexpr double kRamp = 0.5;
  const double ramp = std::min(phaseTime_, kRamp);
  const double sunk = kSinkSpeed * (ramp * ramp / (2.0 * kRamp) + (phaseTime_ - ramp));
  position = segment_.from - Eigen::Vector3d(0, 0, sunk);
  velocity = Eigen::Vector3d(0, 0, -kSinkSpeed * ramp / kRamp);
  acceleration = Eigen::Vector3d(0, 0, phaseTime_ < kRamp ? -kSinkSpeed / kRamp : 0.0);
}

inline void Scene::control() {
  Eigen::Vector3d position, velocity, acceleration;
  const bool wasFlying = phase_ != Phase::kOnPad;
  guide(position, velocity, acceleration);
  referencePosition_ = position;

  const Eigen::Matrix3d orientation = rocket_->getBaseOrientation().e();
  const Eigen::Vector3d axis = orientation.col(2);
  const auto& u = rocket_->getGeneralizedVelocity().e();
  const Eigen::Vector3d omega = u.segment<3>(3);
  const Eigen::Vector3d com = centerOfMass();
  const double mass = rocket_->getTotalMass();
  const Eigen::Vector3d comVelocity = rocket_->getLinearMomentum().e() / mass;

  // Engine cut-off once the legs carry the rocket at the end of the descent.
  if (phase_ == Phase::kTouchdown && onGround()) {
    touchdownSpeed_ = sinkSpeed_;
    landings_++;
    toB_ = !toB_;
    phase_ = Phase::kOnPad;
    phaseTime_ = 0.0;
    engineOn_ = false;
  }
  if (!wasFlying || !engineOn_) {
    for (std::size_t i = 0; i < rocket_->getThrusters().size(); ++i) rocket_->setThrust(i, 0.0);
    gimbalTarget_[gimbalX_ + 1] = gimbalTarget_[gimbalY_ + 1] = 0.0;
    rocket_->setPdTarget(gimbalTarget_, gimbalVelocity_);
    return;
  }

  sinkSpeed_ = -comVelocity.z();

  // Translation: the thrust vector that gives the acceleration of the trajectory
  // plus a correction toward it, tilted at most kMaxTilt from the vertical.
  Eigen::Vector3d force = mass * (acceleration + 1.0 * (position - com) + 1.8 * (velocity - comVelocity) + Eigen::Vector3d(0, 0, kGravity));
  force.z() = std::max(force.z(), 0.2 * mass * kGravity);
  const double horizontal = force.head<2>().norm();
  if (horizontal > std::tan(kMaxTilt) * force.z()) force.head<2>() *= std::tan(kMaxTilt) * force.z() / horizontal;
  const Eigen::Vector3d desiredAxis = force.normalized();
  const double thrust = std::clamp(force.dot(axis), kMinThrust, kMaxThrust);
  rocket_->setThrust(engine_, thrust);

  // Pitch and yaw: turn the axis toward the thrust direction by tilting the
  // engine, whose thrust acts at the gimbal, below the center of mass.
  const Eigen::Vector3d pitchYawRate = omega - omega.dot(axis) * axis;
  const Eigen::Vector3d turn = 850.0 * (9.0 * axis.cross(desiredAxis) - 5.0 * pitchYawRate);
  const Eigen::Vector3d torque = orientation.transpose() * turn;
  raisim::Vec<3> gimbal, thrustAxis;
  rocket_->getThrusterPose(engine_, gimbal, thrustAxis);
  const double arm = std::max((com - gimbal.e()).dot(axis), 0.5);
  gimbalTarget_[gimbalX_ + 1] = std::clamp(-torque.x() / (thrust * arm), -kMaxGimbal, kMaxGimbal);
  gimbalTarget_[gimbalY_ + 1] = std::clamp(-torque.y() / (thrust * arm), -kMaxGimbal, kMaxGimbal);
  rocket_->setPdTarget(gimbalTarget_, gimbalVelocity_);

  // Roll: cancel the swirl of the exhaust and keep the fins on the world axes
  // with the roll thrusters, which only push, so one pair fires at a time.
  const double swirl = rocket_->getThrusters()[engine_].torqueCoefficient * rocket_->getThrust(engine_);
  const double rollTorque = -swirl - 30.0 * (4.0 * rollError() + 4.0 * omega.dot(axis));
  const double pairThrust = std::min(std::abs(rollTorque) / (2.0 * kRollArm), kMaxRollThrust);
  const double left = rollTorque > 0.0 ? pairThrust : 0.0, right = rollTorque < 0.0 ? pairThrust : 0.0;
  rocket_->setThrust(roll_[0], left);
  rocket_->setThrust(roll_[1], left);
  rocket_->setThrust(roll_[2], right);
  rocket_->setThrust(roll_[3], right);
}

}  // namespace raisim_examples::rocket_hop

#endif  // RAISIM_EXAMPLES_INCLUDE_ROCKET_HOP_HPP_
