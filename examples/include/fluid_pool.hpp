// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#ifndef RAISIM_EXAMPLES_INCLUDE_FLUID_POOL_HPP_
#define RAISIM_EXAMPLES_INCLUDE_FLUID_POOL_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "Eigen/Core"
#include "raisim/World.hpp"

namespace raisim_examples::fluid_pool {

constexpr double kTimeStep = 0.001;
constexpr double kCycleDuration = 15.0;
constexpr double kSurface = 0.7;
constexpr double kHalfLength = 2.0, kHalfWidth = 1.2;
constexpr double kWaterDensity = 998.2;
constexpr double kIceDensity = 917.0;
constexpr double kWoodDensity = 500.0;
constexpr double kRaftDensity = 400.0;
constexpr double kLogDensity = 650.0;
constexpr double kSteelDensity = 7850.0;
constexpr double kBrickDensity = 1900.0;
constexpr double kIceSide = 0.3, kCubeSide = 0.3;
constexpr double kRaftThickness = 0.1;
constexpr double kPi = 3.14159265358979323846;

// What a cycle ends with
struct CycleResult {
  // part of the ice block's height above the surface: 1 - 917 / 998 = 8 %
  double ice_above = 0.0;
  // angle between the up direction and the most upright face of the half-dense cube: a cube that
  // floats half submerged turns an edge up, 45 degrees
  double cube_tilt_deg = 0.0;
  // depth of the raft's bottom under the surface: 400 / 998 of 10 cm
  double raft_draft = 0.0;
  // height of the steel ball's center over the floor, its radius once it lies there
  double steel_height = 0.0;
};

// A pool 4 m by 2.4 m with water 0.7 m deep (World::addFluidVolume()) under air
// (World::setAmbientFluid()). Bodies of different densities fall in: the ice block floats with
// 8 % above the water, the half-dense wooden cube turns an edge up, the raft and the log float
// flat, the hollow ball rides high, and the steel ball and the brick sink to the floor. The world
// repeats every 15 seconds.
class Scene {
 public:
  Scene();

  // Invoke under RaisimServer's world mutex immediately before integration.
  void update();
  void step() {
    update();
    world_.integrate();
  }
  raisim::World& world() { return world_; }
  const CycleResult& last_result() const { return last_result_; }
  std::size_t completed_cycles() const { return completed_cycles_; }
  CycleResult measure() const;

 private:
  raisim::SingleBodyObject* add(raisim::SingleBodyObject* body, const std::string& name, const std::string& color,
                                const Eigen::Vector3d& position, double yaw);

  raisim::World world_;
  raisim::SingleBodyObject* ice_ = nullptr;
  raisim::SingleBodyObject* cube_ = nullptr;
  raisim::SingleBodyObject* raft_ = nullptr;
  raisim::SingleBodyObject* steel_ = nullptr;
  raisim::World::Checkpoint checkpoint_;
  double start_time_ = 0.0;
  CycleResult last_result_;
  std::size_t completed_cycles_ = 0;
};

inline Scene::Scene() {
  world_.setTimeStep(kTimeStep);
  world_.addGround(0.0, "floor")->setAppearance("0.55,0.7,0.75,1");
  // the walls of the pool
  const double wall = 0.1, height = 1.0;
  const std::array<Eigen::Vector4d, 4> walls{Eigen::Vector4d(0.0, kHalfWidth + 0.5 * wall, 2.0 * kHalfLength + 2.0 * wall, wall),
                                             Eigen::Vector4d(0.0, -kHalfWidth - 0.5 * wall, 2.0 * kHalfLength + 2.0 * wall, wall),
                                             Eigen::Vector4d(kHalfLength + 0.5 * wall, 0.0, wall, 2.0 * kHalfWidth),
                                             Eigen::Vector4d(-kHalfLength - 0.5 * wall, 0.0, wall, 2.0 * kHalfWidth)};
  for (std::size_t i = 0; i < walls.size(); ++i) {
    auto* box = world_.addBox(walls[i][2], walls[i][3], height, 1.0);
    box->setBodyType(raisim::BodyType::STATIC);
    box->setPosition(walls[i][0], walls[i][1], 0.5 * height);
    box->setName("wall_" + std::to_string(i));
    box->setAppearance("0.85,0.85,0.8,1");
  }

  raisim::FluidVolume pool;
  pool.medium = raisim::FluidMedium::water();
  pool.surfaceHeight = kSurface;
  pool.bottomHeight = 0.0;
  pool.min = {-kHalfLength, -kHalfWidth};
  pool.max = {kHalfLength, kHalfWidth};
  world_.addFluidVolume(pool);
  world_.setAmbientFluid(raisim::FluidMedium::air());

  const double ice = kIceSide * kIceSide * kIceSide;
  ice_ = add(world_.addBox(kIceSide, kIceSide, kIceSide, kIceDensity * ice), "ice", "0.8,0.92,1,1", {-1.4, 0.5, 1.1}, 0.2);
  const double cube = kCubeSide * kCubeSide * kCubeSide;
  cube_ = add(world_.addBox(kCubeSide, kCubeSide, kCubeSide, kWoodDensity * cube), "wooden_cube", "0.75,0.55,0.3,1",
              {-0.7, -0.5, 1.1}, 0.0);
  raft_ = add(world_.addBox(1.0, 0.6, kRaftThickness, kRaftDensity * 0.06), "raft", "0.6,0.4,0.2,1", {0.4, 0.45, 1.0}, 0.3);
  const double logRadius = 0.08, logLength = 0.8;
  const double log = kPi * logRadius * logRadius * (logLength + 4.0 / 3.0 * logRadius);
  auto* logBody = add(world_.addCapsule(logRadius, logLength, kLogDensity * log), "log", "0.5,0.33,0.18,1",
                      {0.6, -0.5, 1.2}, 0.0);
  logBody->setOrientation(Eigen::Matrix3d(Eigen::AngleAxisd(0.5 * kPi - 0.3, Eigen::Vector3d::UnitY())));
  add(world_.addSphere(0.15, 0.3), "hollow_ball", "1,0.3,0.25,1", {1.5, 0.6, 1.2}, 0.0);
  const double steelRadius = 0.06;
  steel_ = add(world_.addSphere(steelRadius, kSteelDensity * 4.0 / 3.0 * kPi * steelRadius * steelRadius * steelRadius),
               "steel_ball", "0.7,0.7,0.75,1", {1.4, -0.6, 1.3}, 0.0);
  add(world_.addBox(0.2, 0.1, 0.06, kBrickDensity * 0.0012), "brick", "0.7,0.3,0.2,1", {-1.5, -0.6, 1.3}, 0.7);

  checkpoint_ = world_.captureCheckpoint();
  start_time_ = world_.getWorldTime();
}

inline raisim::SingleBodyObject* Scene::add(raisim::SingleBodyObject* body, const std::string& name,
                                            const std::string& color, const Eigen::Vector3d& position, double yaw) {
  body->setName(name);
  body->setAppearance(color);
  body->setPosition(position[0], position[1], position[2]);
  body->setOrientation(Eigen::Matrix3d(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())));
  return body;
}

inline CycleResult Scene::measure() const {
  CycleResult result;
  const auto top = [](const raisim::SingleBodyObject* body, double half) {
    const Eigen::Matrix3d rotation = body->getRotationMatrix();
    double up = 0.0;
    for (int i = 0; i < 3; ++i) up += std::abs(rotation(2, i)) * half;
    return body->getPosition()[2] + up;
  };
  result.ice_above = (top(ice_, 0.5 * kIceSide) - kSurface) / kIceSide;
  const Eigen::Matrix3d cube = cube_->getRotationMatrix();
  const double upright = cube.row(2).cwiseAbs().maxCoeff();
  result.cube_tilt_deg = std::acos(std::min(1.0, upright)) * 180.0 / kPi;
  const Eigen::Matrix3d raft = raft_->getRotationMatrix();
  result.raft_draft = kSurface - (raft_->getPosition()[2] - 0.5 * kRaftThickness * std::abs(raft(2, 2)));
  result.steel_height = steel_->getPosition()[2];
  return result;
}

inline void Scene::update() {
  const double time = world_.getWorldTime() - start_time_;
  if (!ice_->getPosition().allFinite() || !cube_->getPosition().allFinite())
    throw std::runtime_error("Non-finite state in the pool");
  if (time >= kCycleDuration - 0.5 * kTimeStep) {
    last_result_ = measure();
    ++completed_cycles_;
    std::string error;
    if (!world_.restoreCheckpoint(checkpoint_, &error)) throw std::runtime_error("Cannot reset the world: " + error);
    start_time_ = world_.getWorldTime();
  }
}

}  // namespace raisim_examples::fluid_pool

#endif  // RAISIM_EXAMPLES_INCLUDE_FLUID_POOL_HPP_
