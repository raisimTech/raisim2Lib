// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#ifndef RAISIM_EXAMPLES_INCLUDE_FLUID_STREAM_HPP_
#define RAISIM_EXAMPLES_INCLUDE_FLUID_STREAM_HPP_

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "raisim/World.hpp"

namespace raisim_examples::fluid_stream {

constexpr double kTimeStep = 0.002;
// the channel: a stony bed between two banks, from kStart to kEnd along x, where the water flows
constexpr double kStart = -4.0, kEnd = 16.0, kHalfWidth = 2.5;
constexpr double kBedFriction = 1.2;
constexpr double kMaxSpeed = 3.0;
constexpr double kPi = 3.14159265358979323846;

// Adds the bed, its banks and the water, of the depth given, flowing along +x at speed
// (World::addFluidVolume() with a current), under air. Returns the index of the water's volume.
inline size_t addChannel(raisim::World& world, double depth, double speed) {
  world.setTimeStep(kTimeStep);
  world.setDefaultMaterial(kBedFriction, 0.0, 0.0);
  world.addGround(0.0, "bed")->setAppearance("0.45,0.42,0.36,1");
  const double length = kEnd - kStart, wall = 0.2, height = depth + 0.4;
  for (const double side : {-1.0, 1.0}) {
    auto* bank = world.addBox(length, wall, height, 1.0);
    bank->setBodyType(raisim::BodyType::STATIC);
    bank->setPosition(0.5 * (kStart + kEnd), side * (kHalfWidth + 0.5 * wall), 0.5 * height);
    bank->setName(side < 0.0 ? "right_bank" : "left_bank");
    bank->setAppearance("0.4,0.5,0.32,1");
  }
  raisim::FluidVolume water;
  water.medium = raisim::FluidMedium::water();
  water.medium.velocity = {speed, 0.0, 0.0};
  water.surfaceHeight = depth;
  water.bottomHeight = 0.0;
  water.min = {kStart, -kHalfWidth};
  water.max = {kEnd, kHalfWidth};
  world.setAmbientFluid(raisim::FluidMedium::air());
  return world.addFluidVolume(water);
}

// Sets the speed of the stream, waking what lies in it.
inline void setStreamSpeed(raisim::World& world, size_t river, double speed) {
  auto water = world.getFluidVolumes()[river];
  if (water.medium.velocity[0] == speed) return;
  water.medium.velocity = {speed, 0.0, 0.0};
  world.setFluidVolume(river, water);
}

// Puts a body at a pose at rest.
inline void place(raisim::SingleBodyObject* body, const Eigen::Vector3d& position, double yaw, double tilt = 0.0) {
  body->setPosition(position.x(), position.y(), position.z());
  body->setOrientation(Eigen::Matrix3d(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                                       Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitX())));
  body->setLinearVelocity(Eigen::Vector3d::Zero());
  body->setAngularVelocity(Eigen::Vector3d::Zero());
}

// Bodies the stream carries down a channel 0.8 m deep, past a post: crates, logs and beach balls
// float off with it, and bricks on the bed stay until the stream, past about 2.1 m/s, slides them.
// What reaches the end of the channel drops back in at its start.
class DebrisScene {
 public:
  explicit DebrisScene(double speed = 0.8) {
    river_ = addChannel(world_, kDepth, speed);
    auto* post = world_.addCylinder(0.15, kDepth + 0.6, 1.0);
    post->setBodyType(raisim::BodyType::STATIC);
    post->setPosition(4.0, 0.6, 0.5 * (kDepth + 0.6));
    post->setName("post");
    post->setAppearance("0.35,0.3,0.25,1");
    const auto add = [&](raisim::SingleBodyObject* body, const std::string& name, const std::string& color,
                         bool onBed) {
      body->setName(name);
      body->setAppearance(color);
      bodies_.push_back({body, onBed});
    };
    for (int i = 0; i < 3; ++i) add(world_.addBox(0.4, 0.4, 0.4, 500.0 * 0.064), "crate", "0.62,0.45,0.25,1", false);
    const double radius = 0.12, length = 1.2;
    const double log = kPi * radius * radius * (length + 4.0 / 3.0 * radius);
    for (int i = 0; i < 2; ++i) add(world_.addCapsule(radius, length, 650.0 * log), "log", "0.45,0.3,0.17,1", false);
    for (int i = 0; i < 2; ++i) add(world_.addSphere(0.25, 2.0), "beach_ball", i == 0 ? "1,0.3,0.2,1" : "1,0.85,0.2,1", false);
    for (int i = 0; i < 3; ++i) add(world_.addBox(0.4, 0.3, 0.2, kBrickDensity * 0.024), "brick", "0.65,0.3,0.22,1", true);
    for (size_t i = 0; i < bodies_.size(); ++i) drop(i, kStart + 1.0 + 1.1 * double(i));
  }

  // Sets the stream and drops what reached the end of the channel back in at its start. Invoke under
  // RaisimServer's world mutex immediately before integration.
  void update(double speed) {
    setStreamSpeed(world_, river_, speed);
    for (size_t i = 0; i < bodies_.size(); ++i)
      if (bodies_[i].first->getPosition()[0] > kEnd - 0.8) {
        drop(i, kStart + 0.8);
        ++recycled_;
      }
  }
  void step(double speed) {
    update(speed);
    world_.integrate();
  }

  raisim::World& world() { return world_; }
  // the bodies and whether each started on the bed
  const std::vector<std::pair<raisim::SingleBodyObject*, bool>>& bodies() const { return bodies_; }
  // how many bodies reached the end of the channel
  size_t recycled() const { return recycled_; }

  static constexpr double kDepth = 0.8;
  static constexpr double kBrickDensity = 1600.0;

 private:
  // drops body i into the stream at x, across the channel by its index
  void drop(size_t i, double x) {
    auto* body = bodies_[i].first;
    const double y = -1.6 + 0.8 * double(i % 5);
    if (bodies_[i].second) {
      place(body, {x, y, 0.12}, 0.3 * double(i));
    } else {
      const bool log = body->getObjectType() == raisim::ObjectType::CAPSULE;
      place(body, {x, y, kDepth + 0.4}, 0.7 * double(i), log ? 0.5 * kPi : 0.2 * double(i));
    }
  }

  raisim::World world_;
  size_t river_ = 0;
  std::vector<std::pair<raisim::SingleBodyObject*, bool>> bodies_;
  size_t recycled_ = 0;
};

// ANYmal stands across a stream 0.55 m deep, the water pushing on its side, its joints holding the
// standing pose. A gentle stream leaves it standing; past about 1.2 m/s it tips over and the stream
// carries it off. When it reaches the end of the channel, or has lain fallen for four seconds, it
// stands up at the start again.
class RobotScene {
 public:
  explicit RobotScene(const std::string& urdf, double speed = 0.5) {
    river_ = addChannel(world_, kDepth, speed);
    robot_ = world_.addArticulatedSystem(urdf);
    robot_->setName("anymal");
    nominal_.resize(robot_->getGeneralizedCoordinateDim());
    nominal_ << 0.0, 0.0, 0.54, std::cos(0.25 * kPi), 0.0, 0.0, std::sin(0.25 * kPi), 0.03, 0.4, -0.8, -0.03, 0.4, -0.8,
        0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
    Eigen::VectorXd pGain = Eigen::VectorXd::Zero(robot_->getDOF()), dGain = pGain;
    pGain.tail(12).setConstant(200.0);
    dGain.tail(12).setConstant(5.0);
    robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    robot_->setPdGains(pGain, dGain);
    robot_->setPdTarget(nominal_, Eigen::VectorXd::Zero(robot_->getDOF()));
    restart();
  }

  // Sets the stream and stands the robot up at the start again when it is due. Invoke under
  // RaisimServer's world mutex immediately before integration.
  void update(double speed) {
    setStreamSpeed(world_, river_, speed);
    const double time = world_.getWorldTime();
    if (!fallen()) stood_ = time;
    if (robot_->getBasePosition()[0] > kEnd - 1.0 || time - stood_ > 4.0) {
      ++falls_;
      restart();
    }
  }
  void step(double speed) {
    update(speed);
    world_.integrate();
  }

  raisim::World& world() { return world_; }
  raisim::ArticulatedSystem* robot() { return robot_; }
  // whether the base is tilted past 60 degrees
  bool fallen() const { return robot_->getBaseOrientation()(2, 2) < 0.5; }
  // how far the stream carried the robot from where it stood
  double drift() const { return robot_->getBasePosition()[0]; }
  // how many times it started over after a fall
  size_t falls() const { return falls_; }

  static constexpr double kDepth = 0.55;

 private:
  void restart() {
    robot_->setState(nominal_, Eigen::VectorXd::Zero(robot_->getDOF()));
    stood_ = world_.getWorldTime();
  }

  raisim::World world_;
  size_t river_ = 0;
  raisim::ArticulatedSystem* robot_ = nullptr;
  Eigen::VectorXd nominal_;
  // the world time at which it last stood
  double stood_ = 0.0;
  size_t falls_ = 0;
};

}  // namespace raisim_examples::fluid_stream

#endif  // RAISIM_EXAMPLES_INCLUDE_FLUID_STREAM_HPP_
