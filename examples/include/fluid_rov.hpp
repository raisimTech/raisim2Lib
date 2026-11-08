// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#ifndef RAISIM_EXAMPLES_INCLUDE_FLUID_ROV_HPP_
#define RAISIM_EXAMPLES_INCLUDE_FLUID_ROV_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "raisim/World.hpp"

namespace raisim_examples::fluid_rov {

constexpr double kTimeStep = 0.001;
constexpr double kSurface = 6.0;
constexpr double kDepth = 3.0;
// the current turns by a quarter every kCurrentPeriod seconds
constexpr double kCurrentSpeed = 0.5;
constexpr double kCurrentPeriod = 8.0;
constexpr double kMaxThrust = 60.0;
constexpr double kMass = 88.0;
constexpr double kPi = 3.14159265358979323846;

// One thruster: where it pushes in the vehicle frame, and along which direction
struct Thruster {
  Eigen::Vector3d position;
  Eigen::Vector3d direction;
};

// A remotely operated vehicle 3 m under the surface of the sea (World::addFluidVolume()), held at
// its station by six thrusters against a current of 0.5 m/s that turns by a quarter every eight
// seconds. The sea holds the vehicle up a little more than its weight, drags it with the current,
// and moves the water around it with it (its added mass); its center of mass lies below its center
// of buoyancy, so it stays level by itself. The thrusters of a proportional-integral-derivative
// controller keep it at its station and heading.
class Scene {
 public:
  Scene();
  explicit Scene(const std::string& scenePath);

  // Invoke under RaisimServer's world mutex immediately before integration; it acts once for each
  // world time, so calls while the world is paused change nothing.
  void update();
  void step() {
    update();
    world_.integrate();
  }
  raisim::World& world() { return world_; }
  raisim::Compound* vehicle() { return vehicle_; }
  const std::vector<Thruster>& thrusters() const { return thrusters_; }
  const Eigen::VectorXd& thrust() const { return thrust_; }
  // distance from the station [m]
  double positionError() const { return (vehicle_->getComPosition() - station_).norm(); }
  // the current at this moment
  Eigen::Vector3d current() const;

 private:
  void initialize(bool detailedVisuals);

  raisim::World world_;
  raisim::Compound* vehicle_ = nullptr;
  size_t sea_ = 0;
  std::vector<Thruster> thrusters_;
  Eigen::MatrixXd allocation_;
  Eigen::VectorXd thrust_;
  Eigen::Vector3d station_{0.0, 0.0, kSurface - kDepth};
  Eigen::Vector3d integral_ = Eigen::Vector3d::Zero();
  double yawIntegral_ = 0.0;
  // the world time of the last update(), negative before the first
  double updated_ = -1.0;
};

inline Scene::Scene() {
  initialize(false);
}

inline Scene::Scene(const std::string& scenePath) : world_(scenePath) {
  initialize(true);
}

inline void Scene::initialize(bool detailedVisuals) {
  world_.setTimeStep(kTimeStep);
  world_.addGround(0.0, "seabed")->setAppearance(detailedVisuals ? "hidden" : "0.55,0.5,0.4,1");

  raisim::FluidVolume sea;
  sea.medium = raisim::FluidMedium::seawater();
  sea.surfaceHeight = kSurface;
  sea_ = world_.addFluidVolume(sea);

  // a hull, a foam block on top and two skids below: the foam makes the vehicle buoyant and the skids
  // put its center of mass under its center of buoyancy
  std::vector<raisim::Compound::CompoundObjectChild> children(4);
  children[0].objectType = raisim::ObjectType::BOX;
  children[0].objectParam = {0.6, 0.45, 0.25, 0.0};
  children[0].appearance = "0.9,0.75,0.15,1";
  children[1].objectType = raisim::ObjectType::BOX;
  children[1].objectParam = {0.55, 0.4, 0.08, 0.0};
  children[1].trans.pos = {0.0, 0.0, 0.165};
  children[1].appearance = "0.95,0.95,0.9,1";
  for (int side = 0; side < 2; ++side) {
    auto& skid = children[2 + side];
    skid.objectType = raisim::ObjectType::CYLINDER;
    skid.objectParam = {0.03, 0.6, 0.0, 0.0};
    skid.trans.pos = {0.0, side == 0 ? 0.18 : -0.18, -0.16};
    skid.trans.rot = raisim::Mat<3, 3>{0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0};
    skid.appearance = "0.2,0.2,0.2,1";
  }
  if (detailedVisuals)
    for (auto& child : children) child.appearance = "hidden";
  raisim::Mat<3, 3> inertia;
  inertia.setZero();
  inertia(0, 0) = kMass / 12.0 * (0.45 * 0.45 + 0.35 * 0.35);
  inertia(1, 1) = kMass / 12.0 * (0.6 * 0.6 + 0.35 * 0.35);
  inertia(2, 2) = kMass / 12.0 * (0.6 * 0.6 + 0.45 * 0.45);
  vehicle_ = world_.addCompound(children, kMass, {0.0, 0.0, -0.05}, inertia);
  vehicle_->setName("rov");
  vehicle_->setPosition(station_[0], station_[1], station_[2] + 0.05);

  // four vectored thrusters in the horizontal plane at the corners, two vertical ones on the sides
  const double s = 1.0 / std::sqrt(2.0);
  thrusters_ = {{{0.3, 0.22, -0.05}, {s, -s, 0.0}},  {{0.3, -0.22, -0.05}, {s, s, 0.0}},
                {{-0.3, 0.22, -0.05}, {s, s, 0.0}},  {{-0.3, -0.22, -0.05}, {s, -s, 0.0}},
                {{0.0, 0.25, 0.0}, {0.0, 0.0, 1.0}}, {{0.0, -0.25, 0.0}, {0.0, 0.0, 1.0}}};
  // the force and torque about the center of mass of each thruster at unit thrust: four rows of x,
  // y, z force and yaw torque
  const Eigen::Vector3d com(0.0, 0.0, -0.05);
  Eigen::MatrixXd effect(4, thrusters_.size());
  for (size_t i = 0; i < thrusters_.size(); ++i) {
    const auto& t = thrusters_[i];
    effect.block<3, 1>(0, i) = t.direction;
    effect(3, i) = (t.position - com).cross(t.direction).z();
  }
  allocation_ = effect.completeOrthogonalDecomposition().pseudoInverse();
  thrust_ = Eigen::VectorXd::Zero(thrusters_.size());
}

inline Eigen::Vector3d Scene::current() const {
  const double heading = 0.5 * kPi * std::floor(world_.getWorldTime() / kCurrentPeriod);
  return kCurrentSpeed * Eigen::Vector3d(std::cos(heading), std::sin(heading), 0.0);
}

inline void Scene::update() {
  // once a step: while the viewer pauses the world, the thrust set for the next step stands, as
  // setExternalForce() adds up until a step
  if (world_.getWorldTime() == updated_) return;
  updated_ = world_.getWorldTime();
  auto sea = world_.getFluidVolumes()[sea_];
  const Eigen::Vector3d flow = current();
  if ((sea.medium.velocity.e() - flow).norm() > 0.0) {
    sea.medium.velocity = {flow[0], flow[1], flow[2]};
    world_.setFluidVolume(sea_, sea);
  }

  const Eigen::Vector3d position = vehicle_->getComPosition();
  const Eigen::Vector3d velocity = vehicle_->getLinearVelocity();
  if (!position.allFinite()) throw std::runtime_error("Non-finite vehicle state");
  const Eigen::Matrix3d rotation = vehicle_->getRotationMatrix();
  const double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
  const double yawRate = vehicle_->getAngularVelocity()[2];

  // a proportional-integral-derivative controller of the station and the heading, in the world frame
  const Eigen::Vector3d error = station_ - position;
  integral_ += kTimeStep * error;
  yawIntegral_ -= kTimeStep * yaw;
  const Eigen::Vector3d force = 400.0 * error + 120.0 * integral_ - 300.0 * velocity;
  const double torque = -60.0 * yaw + 20.0 * yawIntegral_ - 40.0 * yawRate;
  Eigen::Vector4d wanted;
  wanted << rotation.transpose() * force, torque;
  thrust_ = (allocation_ * wanted).cwiseMax(-kMaxThrust).cwiseMin(kMaxThrust);
  for (size_t i = 0; i < thrusters_.size(); ++i) {
    const Eigen::Vector3d push = rotation * (thrust_[i] * thrusters_[i].direction);
    vehicle_->setExternalForce(0, {thrusters_[i].position[0], thrusters_[i].position[1], thrusters_[i].position[2]},
                               {push[0], push[1], push[2]});
  }
}

}  // namespace raisim_examples::fluid_rov

#endif  // RAISIM_EXAMPLES_INCLUDE_FLUID_ROV_HPP_
