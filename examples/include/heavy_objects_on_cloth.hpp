// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#ifndef RAISIM_EXAMPLES_INCLUDE_HEAVY_OBJECTS_ON_CLOTH_HPP_
#define RAISIM_EXAMPLES_INCLUDE_HEAVY_OBJECTS_ON_CLOTH_HPP_

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "raisim/World.hpp"

namespace raisim_examples::heavy_cloth {

constexpr double kTimeStep = 0.001;
constexpr double kCycleDuration = 5.0;
constexpr double kClothRadius = 0.008;

// How far each load presses into what carries it, over the last second of a
// cycle, when everything has settled, and where the loads rest at its end.
struct CycleResult {
  double ball_cloth_in_ground = 0.0;     // particles under the 10 kg ball, below the ground
  double layers_overlap = 0.0;           // particles of the upper cloth inside those of the lower one
  double blanket_box_in_ground = 0.0;    // 1 kg box under the 5 kg blanket, below the ground
  double ball_height = 0.0;              // center of the ball
  double box_height = 0.0;               // center of the 5 kg box on the layers
  double blanket_top = 0.0;              // highest particle of the blanket
};

// Three loads on cloth, solved together with the bodies
// (World::setMonolithicDeformables()): a 10 kg ball dropped on a cloth on the
// ground, a 5 kg box dropped on two stacked cloths, and a 5 kg blanket dropped
// over a 1 kg box. The world repeats from a checkpoint every five seconds.
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
  raisim::DeformableObject* AddCloth(int n, double width, const raisim::Vec<3>& center, double mass, const std::string& name);
  void Observe();

  raisim::World world_;
  raisim::DeformableObject* ball_cloth_ = nullptr;
  raisim::DeformableObject* lower_layer_ = nullptr;
  raisim::DeformableObject* upper_layer_ = nullptr;
  raisim::DeformableObject* blanket_ = nullptr;
  raisim::SingleBodyObject* ball_ = nullptr;
  raisim::SingleBodyObject* layered_box_ = nullptr;
  raisim::SingleBodyObject* blanket_box_ = nullptr;
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
  world_.setDefaultMaterial(0.5, 0.0, 0.01);
  world_.addGround(0.0, "ground")->setAppearance("checkerboard");

  ball_cloth_ = AddCloth(33, 0.64, {-0.9, 0.0, kClothRadius + 0.001}, 0.2, "cloth_under_ball");
  ball_ = world_.addSphere(0.08, 10.0);
  ball_->setName("ball_10kg");
  ball_->setPosition(-0.9, 0.0, 0.35);
  ball_->setAppearance("0.85,0.55,0.2,1");

  lower_layer_ = AddCloth(17, 0.32, {0.0, 0.0, kClothRadius + 0.001}, 0.1, "lower_layer");
  upper_layer_ = AddCloth(17, 0.32, {0.003, 0.002, 3.0 * kClothRadius + 0.002}, 0.1, "upper_layer");
  layered_box_ = world_.addBox(0.15, 0.15, 0.05, 5.0);
  layered_box_->setName("box_5kg");
  layered_box_->setPosition(0.0, 0.0, 0.25);
  layered_box_->setAppearance("0.35,0.35,0.38,1");

  blanket_box_ = world_.addBox(0.3, 0.3, 0.2, 1.0);
  blanket_box_->setName("box_1kg");
  blanket_box_->setPosition(0.9, 0.0, 0.1);
  blanket_box_->setAppearance("0.3,0.6,0.35,1");
  blanket_ = AddCloth(25, 0.5, {0.9, 0.0, 0.3}, 5.0, "blanket_5kg");

  checkpoint_ = world_.captureCheckpoint();
  start_time_ = world_.getWorldTime();
}

inline raisim::DeformableObject* Scene::AddCloth(int n, double width, const raisim::Vec<3>& center, double mass, const std::string& name) {
  std::vector<raisim::Vec<3>> vertices;
  std::vector<raisim::DeformableObject::Triangle> triangles;
  const double spacing = width / (n - 1);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      vertices.push_back(raisim::Vec<3>{center[0] - 0.5 * width + spacing * i, center[1] - 0.5 * width + spacing * j, center[2]});
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
  material.totalMass = mass;
  material.distanceCompliance = 1.0e-6;
  material.bendStiffness = 1.0e-4;
  material.collisionRadius = kClothRadius;
  material.damping = 0.002;
  auto* cloth = world_.addDeformableCloth(vertices, triangles, {}, material);
  cloth->setName(name);
  return cloth;
}

inline void Scene::Update() {
  Observe();
  if (world_.getWorldTime() - start_time_ >= kCycleDuration - 0.5 * kTimeStep) {
    last_result_ = current_;
    ++completed_cycles_;
    std::string error;
    if (!world_.restoreCheckpoint(checkpoint_, &error)) throw std::runtime_error("Cannot reset the world: " + error);
    start_time_ = world_.getWorldTime();
    current_ = CycleResult{};
  }
}

inline void Scene::Observe() {
  const auto finite = [](const raisim::DeformableObject* cloth) {
    for (const auto& p : cloth->getPositions()) {
      if (!p.e().allFinite()) throw std::runtime_error("Non-finite state of " + cloth->getName());
    }
  };
  for (const auto* cloth : {ball_cloth_, lower_layer_, upper_layer_, blanket_}) finite(cloth);
  for (const auto* body : {ball_, layered_box_, blanket_box_}) {
    if (!body->getPosition().allFinite()) throw std::runtime_error("Non-finite state of " + body->getName());
  }
  if (world_.getWorldTime() - start_time_ < kCycleDuration - 1.0) return;
  for (const auto& p : ball_cloth_->getPositions()) current_.ball_cloth_in_ground = std::max(current_.ball_cloth_in_ground, kClothRadius - p[2]);
  for (const auto& upper : upper_layer_->getPositions()) {
    for (const auto& lower : lower_layer_->getPositions()) {
      current_.layers_overlap = std::max(current_.layers_overlap, 2.0 * kClothRadius - (upper.e() - lower.e()).norm());
    }
  }
  current_.blanket_box_in_ground = std::max(current_.blanket_box_in_ground, 0.1 - blanket_box_->getPosition()[2]);
  current_.ball_height = ball_->getPosition()[2];
  current_.box_height = layered_box_->getPosition()[2];
  current_.blanket_top = -1.0;
  for (const auto& p : blanket_->getPositions()) current_.blanket_top = std::max(current_.blanket_top, p[2]);
}

}  // namespace raisim_examples::heavy_cloth

#endif  // RAISIM_EXAMPLES_INCLUDE_HEAVY_OBJECTS_ON_CLOTH_HPP_
