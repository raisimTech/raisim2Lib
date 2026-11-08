// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#ifndef RAISIM_EXAMPLES_INCLUDE_CLOTH_ON_SLOPE_HPP_
#define RAISIM_EXAMPLES_INCLUDE_CLOTH_ON_SLOPE_HPP_

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "raisim/World.hpp"

namespace raisim_examples::cloth_slope {

constexpr double kTimeStep = 0.001;
constexpr double kCycleDuration = 4.0;
constexpr double kFriction = 0.5;
constexpr double kClothRadius = 0.008;
// Below the friction angle (tan 18 deg = 0.32 < 0.5) the cloth and the box stay
// put; above it (tan 30 deg = 0.58) they slide down together.
constexpr std::array<double, 2> kRampAngles{0.31415926536, 0.52359877560};

// How far each box moved along its ramp from 0.5 s to 1.5 s into a cycle
struct CycleResult {
  std::array<double, 2> box_travel{0.0, 0.0};
};

// A 1 kg box on a cloth on each of two static ramps, solved together with the
// cloth (World::setMonolithicDeformables()). The ground holds the cloth under
// the box with the friction of the box's weight, so on the gentle ramp both
// stay; on the steep one they slide. The world repeats every four seconds.
class Scene {
 public:
  explicit Scene(bool monolithic = true);

  // Invoke under RaisimServer's world mutex immediately before integration.
  void Update();
  void Step() {
    Update();
    world_.integrate();
  }
  raisim::World& world() { return world_; }
  const CycleResult& last_result() const { return last_result_; }
  std::size_t completed_cycles() const { return completed_cycles_; }

 private:
  void AddRamp(std::size_t index, double y);

  raisim::World world_;
  std::array<raisim::SingleBodyObject*, 2> boxes_{};
  std::array<Eigen::Vector3d, 2> downhill_{};
  std::array<Eigen::Vector3d, 2> box_start_{};
  raisim::World::Checkpoint checkpoint_;
  double start_time_ = 0.0;
  CycleResult current_;
  CycleResult last_result_;
  std::size_t completed_cycles_ = 0;
};

inline Scene::Scene(bool monolithic) {
  world_.setMonolithicDeformables(monolithic);
  world_.setTimeStep(kTimeStep);
  world_.setSleepingEnabled(false);
  world_.setDefaultMaterial(kFriction, 0.0, 0.01);
  world_.addGround(-0.2, "ground")->setAppearance("checkerboard");
  AddRamp(0, -0.45);
  AddRamp(1, 0.45);
  checkpoint_ = world_.captureCheckpoint();
  start_time_ = world_.getWorldTime();
}

inline void Scene::AddRamp(std::size_t index, double y) {
  const double angle = kRampAngles[index];
  const double thickness = 0.05, length = 1.6;
  // the ramp descends toward +x: its top normal is (sin a, 0, cos a)
  const Eigen::Vector3d along(std::cos(angle), 0.0, -std::sin(angle));
  const Eigen::Vector3d normal(std::sin(angle), 0.0, std::cos(angle));
  const Eigen::Vector3d center(0.0, y, 0.5 * length * std::sin(angle));
  const raisim::Vec<4> orientation{std::cos(0.5 * angle), 0.0, std::sin(0.5 * angle), 0.0};

  auto* ramp = world_.addBox(length, 0.6, thickness, 1.0);
  ramp->setName("ramp_" + std::to_string(index));
  ramp->setBodyType(raisim::BodyType::STATIC);
  ramp->setPosition(center[0], center[1], center[2]);
  ramp->setOrientation(orientation);
  ramp->setAppearance("0.45,0.42,0.38,1");

  // a 0.48 m square cloth on the upper part of the ramp
  const int n = 25;
  const double width = 0.48, spacing = width / (n - 1);
  const Eigen::Vector3d clothCenter = center - 0.35 * along + (0.5 * thickness + kClothRadius + 0.001) * normal;
  std::vector<raisim::Vec<3>> vertices;
  std::vector<raisim::DeformableObject::Triangle> triangles;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const Eigen::Vector3d p = clothCenter + (spacing * i - 0.5 * width) * along + Eigen::Vector3d(0.0, spacing * j - 0.5 * width, 0.0);
      vertices.push_back(raisim::Vec<3>{p[0], p[1], p[2]});
    }
  }
  for (int j = 0; j + 1 < n; ++j) {
    for (int i = 0; i + 1 < n; ++i) {
      const auto a = static_cast<unsigned>(j * n + i);
      const unsigned b = a + 1, c = a + n, d = c + 1;
      triangles.push_back({a, b, d});
      triangles.push_back({a, d, c});
    }
  }
  raisim::DeformableObject::Material material;
  material.totalMass = 0.15;
  material.distanceCompliance = 1.0e-6;
  material.bendStiffness = 1.0e-4;
  material.collisionRadius = kClothRadius;
  material.damping = 0.002;
  world_.addDeformableCloth(vertices, triangles, {}, material)->setName("cloth_" + std::to_string(index));

  const Eigen::Vector3d boxCenter = clothCenter + (kClothRadius + 0.001 + 0.025) * normal;
  auto* box = world_.addBox(0.12, 0.12, 0.05, 1.0);
  box->setName("box_" + std::to_string(index));
  box->setPosition(boxCenter[0], boxCenter[1], boxCenter[2]);
  box->setOrientation(orientation);
  box->setAppearance(index == 0 ? "0.3,0.6,0.35,1" : "0.85,0.35,0.25,1");
  boxes_[index] = box;
  downhill_[index] = along;
}

inline void Scene::Update() {
  const double time = world_.getWorldTime() - start_time_;
  for (std::size_t i = 0; i < boxes_.size(); ++i) {
    const Eigen::Vector3d position = boxes_[i]->getPosition();
    if (!position.allFinite()) throw std::runtime_error("Non-finite box state");
    if (std::abs(time - 0.5) < 0.5 * kTimeStep) box_start_[i] = position;
    if (time > 0.5 && time < 1.5 + 0.5 * kTimeStep) current_.box_travel[i] = (position - box_start_[i]).dot(downhill_[i]);
  }
  if (time >= kCycleDuration - 0.5 * kTimeStep) {
    last_result_ = current_;
    ++completed_cycles_;
    std::string error;
    if (!world_.restoreCheckpoint(checkpoint_, &error)) throw std::runtime_error("Cannot reset the world: " + error);
    start_time_ = world_.getWorldTime();
    current_ = CycleResult{};
  }
}

}  // namespace raisim_examples::cloth_slope

#endif  // RAISIM_EXAMPLES_INCLUDE_CLOTH_ON_SLOPE_HPP_
