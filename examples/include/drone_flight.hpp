// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#ifndef RAISIM_EXAMPLES_INCLUDE_DRONE_FLIGHT_HPP_
#define RAISIM_EXAMPLES_INCLUDE_DRONE_FLIGHT_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "raisim/World.hpp"

// A quadrotor drone takes off from its pad, flies figure eights around two
// posts, always facing the way it flies, lands and takes off again. Four URDF
// thrusters in the plus layout fly it: the front and back propellers spin one
// way and the left and right ones the other, so the difference of their
// reaction torques turns the drone about its vertical axis.
namespace raisim_examples::drone_flight {

constexpr double kTimeStep = 0.001;
constexpr double kGravity = 9.81;
constexpr double kPi = 3.14159265358979323846;

// The drone: propellers kArm from the center, each with a reaction torque of
// kDrag times its thrust, and skids kSkidDepth below the base frame [m].
constexpr double kMass = 1.5;
constexpr double kArm = 0.2;
constexpr double kDrag = 0.016;
constexpr double kMaxThrust = 15.0;
constexpr double kSkidDepth = 0.08;

// The figure eight x = kHalfLength sin(a), y = kHalfWidth sin(2a) at
// kCruiseHeight, one loop every kLoopTime, around posts at x = +-kPostX.
constexpr double kCruiseHeight = 1.5;
constexpr double kHalfLength = 2.5;
constexpr double kHalfWidth = 1.0;
constexpr double kLoopTime = 14.0;
constexpr int kLoops = 2;
constexpr double kPostX = 1.75;
constexpr double kPostRadius = 0.1;
constexpr double kPostHeight = 2.5;

constexpr double kRestTime = 2.0;
constexpr double kTakeoffTime = 3.0;
// The drone speeds up into the figure eight and slows down out of it in this time [s].
constexpr double kRampTime = 4.0;
constexpr double kLandingTime = 4.0;
// The landing aims this far below the resting height, so the skids press on the pad [m].
constexpr double kLandingPress = 0.03;

// Position gains per unit mass [1/s^2, 1/s] and attitude gains about the body axes [N m/rad, N m s/rad].
constexpr double kPositionP = 6.0;
constexpr double kPositionD = 4.5;
constexpr std::array<double, 3> kAttitudeP{2.0, 2.0, 0.8};
constexpr std::array<double, 3> kAttitudeD{0.3, 0.3, 0.2};

/// The drone's URDF: a body, two crossed arms, a propeller disc at the end of
/// each arm (the front one orange) and two skids.
inline std::string droneUrdf() {
  const auto material = [](const std::string& name, const std::string& rgba) {
    return "<material name=\"" + name + "\"><color rgba=\"" + rgba + "\"/></material>";
  };
  const std::string dark = material("frame", "0.16 0.17 0.19 1");
  const auto box = [](const std::string& xyz, const std::string& size, const std::string& look, bool collides) {
    const std::string geometry = "<origin xyz=\"" + xyz + "\"/><geometry><box size=\"" + size + "\"/></geometry>";
    return "<visual>" + geometry + look + "</visual>" + (collides ? "<collision>" + geometry + "</collision>" : "");
  };
  std::string urdf =
      "<robot name=\"drone\"><link name=\"base\">"
      "<inertial><mass value=\"" + std::to_string(kMass) + "\"/>"
      "<inertia ixx=\"0.01\" iyy=\"0.01\" izz=\"0.02\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/></inertial>" +
      box("0 0 0", "0.16 0.1 0.06", dark, true) + box("0 0 0.01", "0.44 0.03 0.015", dark, true) +
      box("0 0 0.01", "0.03 0.44 0.015", dark, true);
  for (const double y : {-0.09, 0.09}) {
    const std::string side = std::to_string(y);
    urdf += box("0 " + side + " -0.0725", "0.3 0.015 0.015", dark, true);
    for (const double x : {-0.08, 0.08}) urdf += box(std::to_string(x) + " " + side + " -0.04", "0.012 0.012 0.055", dark, false);
  }
  const char* names[4] = {"front", "back", "left", "right"};
  const double x[4] = {kArm, -kArm, 0.0, 0.0}, y[4] = {0.0, 0.0, kArm, -kArm};
  std::string thrusters;
  for (int i = 0; i < 4; ++i) {
    const std::string at = std::to_string(x[i]) + " " + std::to_string(y[i]) + " 0.03";
    const std::string look = i == 0 ? material("front_rotor", "1.0 0.45 0.1 0.8") : material("rotor", "0.75 0.77 0.8 0.6");
    urdf += "<visual><origin xyz=\"" + at + "\"/><geometry><cylinder radius=\"0.1\" length=\"0.006\"/></geometry>" +
            look + "</visual>";
    thrusters += std::string("<thruster name=\"") + names[i] + "\" link=\"base\" torque_coefficient=\"" +
                 std::to_string(i < 2 ? kDrag : -kDrag) + "\"><origin xyz=\"" + at + "\"/><limit lower=\"0\" upper=\"" +
                 std::to_string(kMaxThrust) + "\"/></thruster>";
  }
  return urdf + "</link>" + thrusters + "</robot>";
}

/// Thrusts of the front, back, left and right propellers that give the total
/// thrust and the body torques (roll, pitch, yaw).
inline Eigen::Vector4d mix(double thrust, const Eigen::Vector3d& torque) {
  const double share = 0.25 * thrust, yaw = torque.z() / (4.0 * kDrag);
  return {share - torque.y() / (2.0 * kArm) + yaw, share + torque.y() / (2.0 * kArm) + yaw,
          share + torque.x() / (2.0 * kArm) - yaw, share - torque.x() / (2.0 * kArm) - yaw};
}

/// A point on the figure eight at angle `angle`, and its derivatives with respect to the angle.
inline void eight(double angle, Eigen::Vector3d& point, Eigen::Vector3d& tangent, Eigen::Vector3d& curvature) {
  point = {kHalfLength * std::sin(angle), kHalfWidth * std::sin(2.0 * angle), kCruiseHeight};
  tangent = {kHalfLength * std::cos(angle), 2.0 * kHalfWidth * std::cos(2.0 * angle), 0.0};
  curvature = {-kHalfLength * std::sin(angle), -4.0 * kHalfWidth * std::sin(2.0 * angle), 0.0};
}

/// What the drone is doing.
enum class Phase { kOnPad, kTakeoff, kEight, kLanding };

/// The drone, its pad and posts, and the flight controller.
class Scene {
 public:
  Scene();

  /// Runs the guidance and control and integrates one step.
  void step();

  raisim::World& world() { return world_; }
  raisim::ArticulatedSystem* drone() const { return drone_; }
  Phase phase() const { return phase_; }
  /// Position and heading the controller follows [m, rad].
  const Eigen::Vector3d& reference() const { return referencePosition_; }
  double referenceYaw() const { return referenceYaw_; }
  /// Heading of the drone's front arm [rad].
  double yaw() const {
    const Eigen::Matrix3d rotation = drone_->getBaseOrientation().e();
    return std::atan2(rotation(1, 0), rotation(0, 0));
  }
  /// The number of landings so far.
  int landings() const { return landings_; }

 private:
  void advancePhase();
  void guide(Eigen::Vector3d& position, Eigen::Vector3d& velocity, Eigen::Vector3d& acceleration, double& yaw,
             double& yawRate);
  void control(const Eigen::Vector3d& position, const Eigen::Vector3d& velocity, const Eigen::Vector3d& acceleration,
               double yaw, double yawRate);
  void eightAngle(double time, double& angle, double& rate, double& acceleration) const;
  bool onGround() const;

  raisim::World world_;
  raisim::ArticulatedSystem* drone_ = nullptr;
  std::size_t ground_ = 0;
  Phase phase_ = Phase::kOnPad;
  double phaseTime_ = 0.0;
  int landings_ = 0;
  Eigen::Vector3d from_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d referencePosition_ = Eigen::Vector3d::Zero();
  double referenceYaw_ = 0.0;
};

inline Scene::Scene() {
  world_.setTimeStep(kTimeStep);
  ground_ = world_.addGround(0.0, "concrete")->getIndexInWorld();
  for (const double x : {-kPostX, kPostX}) {
    auto* post = world_.addCylinder(kPostRadius, kPostHeight, 1.0, "concrete");
    post->setName(x < 0.0 ? "left_post" : "right_post");
    post->setBodyType(raisim::BodyType::STATIC);
    post->setPosition(x, 0.0, 0.5 * kPostHeight);
    post->setAppearance("0.85,0.3,0.2,1");
  }

  drone_ = world_.addArticulatedSystem(droneUrdf());
  drone_->setName("drone");
  Eigen::Vector3d point, tangent, curvature;
  eight(0.0, point, tangent, curvature);
  referenceYaw_ = std::atan2(tangent.y(), tangent.x());
  const Eigen::Quaterniond heading(Eigen::AngleAxisd(referenceYaw_, Eigen::Vector3d::UnitZ()));
  Eigen::VectorXd gc(7);
  gc << 0.0, 0.0, kSkidDepth, heading.w(), heading.x(), heading.y(), heading.z();
  drone_->setGeneralizedCoordinate(gc);
  referencePosition_ = gc.head<3>();
}

inline bool Scene::onGround() const {
  for (const auto& contact : drone_->getContacts())
    if (contact.getPairObjectIndex() == ground_) return true;
  return false;
}

inline void Scene::step() {
  phaseTime_ += kTimeStep;
  advancePhase();
  if (phase_ == Phase::kOnPad) {
    drone_->setThrusts(Eigen::Vector4d::Zero());
  } else {
    Eigen::Vector3d position, velocity, acceleration;
    double yaw = 0.0, yawRate = 0.0;
    guide(position, velocity, acceleration, yaw, yawRate);
    control(position, velocity, acceleration, yaw, yawRate);
  }
  world_.integrate();
}

inline void Scene::advancePhase() {
  const auto enter = [this](Phase next) {
    phase_ = next;
    phaseTime_ = 0.0;
  };
  switch (phase_) {
    case Phase::kOnPad:
      if (phaseTime_ < kRestTime) return;
      from_ = drone_->getBasePosition().e();
      enter(Phase::kTakeoff);
      return;
    case Phase::kTakeoff:
      if (phaseTime_ >= kTakeoffTime) enter(Phase::kEight);
      return;
    case Phase::kEight:
      if (phaseTime_ < kLoops * kLoopTime + kRampTime) return;
      from_ = referencePosition_;
      enter(Phase::kLanding);
      return;
    case Phase::kLanding:
      // The motors stop once the skids are on the ground.
      if (phaseTime_ < kLandingTime || !onGround()) return;
      landings_++;
      enter(Phase::kOnPad);
      return;
  }
}

inline void Scene::eightAngle(double time, double& angle, double& rate, double& acceleration) const {
  // The angular rate rises smoothly from zero to its cruise value in kRampTime
  // and falls back to zero at the end, so the drone enters and leaves the
  // figure eight at rest, over the pad.
  const double cruise = 2.0 * kPi / kLoopTime;
  const double total = kLoops * kLoopTime + kRampTime;
  const double fromEnd = total - time;
  const double ramp = std::min({time, fromEnd, kRampTime});
  const double u = std::clamp(ramp / kRampTime, 0.0, 1.0);
  const double covered = u * u * u * u * (2.5 - 3.0 * u + u * u);
  rate = cruise * u * u * u * (10.0 - 15.0 * u + 6.0 * u * u);
  acceleration = (time < fromEnd ? 1.0 : -1.0) * cruise * 30.0 * u * u * (1.0 - u) * (1.0 - u) / kRampTime;
  if (time < kRampTime)
    angle = cruise * kRampTime * covered;
  else if (fromEnd < kRampTime)
    angle = 2.0 * kPi * kLoops - cruise * kRampTime * covered;
  else
    angle = cruise * (time - 0.5 * kRampTime);
}

inline void Scene::guide(Eigen::Vector3d& position, Eigen::Vector3d& velocity, Eigen::Vector3d& acceleration,
                         double& yaw, double& yawRate) {
  Eigen::Vector3d point, tangent, curvature;
  eight(0.0, point, tangent, curvature);
  yaw = std::atan2(tangent.y(), tangent.x());
  yawRate = 0.0;
  if (phase_ == Phase::kEight) {
    double angle = 0.0, rate = 0.0, angularAcceleration = 0.0;
    eightAngle(phaseTime_, angle, rate, angularAcceleration);
    eight(angle, point, tangent, curvature);
    position = point;
    velocity = rate * tangent;
    acceleration = rate * rate * curvature + angularAcceleration * tangent;
    yaw = std::atan2(tangent.y(), tangent.x());
    yawRate = rate * (tangent.x() * curvature.y() - tangent.y() * curvature.x()) / tangent.squaredNorm();
  } else {
    const bool takeoff = phase_ == Phase::kTakeoff;
    const Eigen::Vector3d to = takeoff ? point : Eigen::Vector3d(0.0, 0.0, kSkidDepth - kLandingPress);
    const double duration = takeoff ? kTakeoffTime : kLandingTime;
    const double s = std::clamp(phaseTime_ / duration, 0.0, 1.0);
    const Eigen::Vector3d delta = to - from_;
    position = from_ + s * s * s * (10.0 - 15.0 * s + 6.0 * s * s) * delta;
    velocity = 30.0 * s * s * (1.0 - s) * (1.0 - s) / duration * delta;
    acceleration = 60.0 * s * (1.0 - s) * (1.0 - 2.0 * s) / (duration * duration) * delta;
  }
  referencePosition_ = position;
  referenceYaw_ = yaw;
}

inline void Scene::control(const Eigen::Vector3d& position, const Eigen::Vector3d& velocity,
                           const Eigen::Vector3d& acceleration, double yaw, double yawRate) {
  // The force the propellers must give, and the attitude that points their
  // thrust along it with the front arm toward the heading.
  const Eigen::Matrix3d rotation = drone_->getBaseOrientation().e();
  const Eigen::VectorXd& u = drone_->getGeneralizedVelocity().e();
  const Eigen::Vector3d force =
      kMass * (acceleration + kPositionP * (position - drone_->getBasePosition().e()) +
               kPositionD * (velocity - u.head<3>()) + kGravity * Eigen::Vector3d::UnitZ());
  Eigen::Matrix3d desired;
  desired.col(2) = force.normalized();
  desired.col(1) = desired.col(2).cross(Eigen::Vector3d(std::cos(yaw), std::sin(yaw), 0.0)).normalized();
  desired.col(0) = desired.col(1).cross(desired.col(2));
  const Eigen::Matrix3d mismatch = desired.transpose() * rotation - rotation.transpose() * desired;
  const Eigen::Vector3d attitudeError = 0.5 * Eigen::Vector3d(mismatch(2, 1), mismatch(0, 2), mismatch(1, 0));
  const Eigen::Vector3d rateError = rotation.transpose() * (u.segment<3>(3) - yawRate * Eigen::Vector3d::UnitZ());
  Eigen::Vector3d torque;
  for (int axis = 0; axis < 3; ++axis)
    torque[axis] = -kAttitudeP[axis] * attitudeError[axis] - kAttitudeD[axis] * rateError[axis];
  drone_->setThrusts(mix(force.dot(rotation.col(2)), torque));
}

}  // namespace raisim_examples::drone_flight

#endif  // RAISIM_EXAMPLES_INCLUDE_DRONE_FLIGHT_HPP_
