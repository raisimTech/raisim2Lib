// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#ifndef RAISIM_EXAMPLES_INCLUDE_TABLECLOTH_PULL_HPP_
#define RAISIM_EXAMPLES_INCLUDE_TABLECLOTH_PULL_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "raisim/World.hpp"

namespace raisim_examples::tablecloth {

constexpr double kTimeStep = 0.001;
constexpr double kCycleDuration = 3.0;
constexpr double kTableTop = 0.75;
constexpr double kTableEdge = -0.45;  // x of the table edge the cloth is pulled over
constexpr double kClothRadius = 0.008;
constexpr double kPullStart = 1.0;
constexpr double kPullDuration = 0.15;
constexpr double kPullForce = 60.0;  // N, shared by the particles of the pulled edge
constexpr int kClothColumns = 41;
constexpr int kClothRows = 33;

// The dishes' largest horizontal displacement in a cycle, and whether the
// cloth left the table top.
struct CycleResult {
  double dish_shift = 0.0;
  bool dishes_on_table = true;
  bool cloth_off_table = false;
};

// The tablecloth trick, with the cloth solved together with the dishes and the
// table (World::setMonolithicDeformables()): three dishes stand on a cloth on a
// table, and a sharp pull on the cloth's edge whips it out from under them.
// Kinetic friction acts on the dishes for the fraction of a second the cloth
// slides under them, too briefly to take them along. The world repeats every
// three seconds.
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
  void Observe();

  raisim::World world_;
  raisim::DeformableObject* cloth_ = nullptr;
  std::array<raisim::SingleBodyObject*, 3> dishes_{};
  std::array<raisim::Vec<3>, 3> dish_start_{};
  // the step the pull was last applied to, so that callbacks without a step (a paused viewer) add none
  double pulled_at_ = -1.0;
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
  world_.addGround(0.0, "floor")->setAppearance("checkerboard");
  world_.setMaterialPairProp("table", "cloth", 0.3, 0.0, 0.01);
  world_.setMaterialPairProp("cloth", "dish", 0.3, 0.0, 0.01);
  world_.setMaterialPairProp("table", "dish", 0.5, 0.0, 0.01);

  auto* table = world_.addBox(0.9, 0.7, 0.05, 1.0, "table");
  table->setName("table");
  table->setBodyType(raisim::BodyType::STATIC);
  table->setPosition(0.0, 0.0, kTableTop - 0.025);
  table->setAppearance("0.55,0.38,0.24,1");
  for (double x : {-0.4, 0.4}) {
    for (double y : {-0.3, 0.3}) {
      auto* leg = world_.addBox(0.04, 0.04, kTableTop - 0.05, 1.0);
      leg->setBodyType(raisim::BodyType::STATIC);
      leg->setPosition(x, y, 0.5 * (kTableTop - 0.05));
      leg->setAppearance("0.4,0.27,0.17,1");
    }
  }

  // a 0.8 m by 0.64 m cloth on the table, its pulled edge 3 cm past the table edge
  std::vector<raisim::Vec<3>> vertices;
  std::vector<raisim::DeformableObject::Triangle> triangles;
  const double width = 0.8, depth = 0.64;
  for (int j = 0; j < kClothRows; ++j) {
    for (int i = 0; i < kClothColumns; ++i) {
      vertices.push_back(raisim::Vec<3>{kTableEdge - 0.03 + width * i / (kClothColumns - 1), -0.5 * depth + depth * j / (kClothRows - 1),
                                        kTableTop + kClothRadius + 0.001});
    }
  }
  for (int j = 0; j + 1 < kClothRows; ++j) {
    for (int i = 0; i + 1 < kClothColumns; ++i) {
      const auto a = static_cast<unsigned>(j * kClothColumns + i);
      const unsigned b = a + 1, c = a + kClothColumns, d = c + 1;
      triangles.push_back({a, b, d});
      triangles.push_back({a, d, c});
    }
  }
  raisim::DeformableObject::Material material;
  material.totalMass = 0.2;
  material.distanceCompliance = 1.0e-6;
  material.bendStiffness = 1.0e-4;
  material.collisionRadius = kClothRadius;
  material.damping = 0.002;
  material.airDrag = 0.5;
  cloth_ = world_.addDeformableCloth(vertices, triangles, {}, material, "cloth");
  cloth_->setName("tablecloth");

  const std::array<std::array<double, 2>, 3> placements{{{-0.05, -0.15}, {0.12, 0.08}, {-0.08, 0.18}}};
  const std::array<const char*, 3> colors{"0.92,0.92,0.88,1", "0.75,0.82,0.9,1", "0.9,0.78,0.6,1"};
  for (std::size_t k = 0; k < dishes_.size(); ++k) {
    auto* dish = world_.addCylinder(0.07, 0.04, 0.6, "dish");
    dish->setName("dish_" + std::to_string(k));
    dish->setPosition(placements[k][0], placements[k][1], kTableTop + 2.0 * kClothRadius + 0.002 + 0.02);
    dish->setAppearance(colors[k]);
    dishes_[k] = dish;
  }

  // let the dishes settle on the cloth
  for (int i = 0; i < 300; ++i) world_.integrate();
  checkpoint_ = world_.captureCheckpoint();
  start_time_ = world_.getWorldTime();
  for (std::size_t k = 0; k < dishes_.size(); ++k) dish_start_[k] = dishes_[k]->getPosition();
}

inline void Scene::Update() {
  Observe();
  const double time = world_.getWorldTime() - start_time_;
  if (time >= kCycleDuration - 0.5 * kTimeStep) {
    last_result_ = current_;
    ++completed_cycles_;
    std::string error;
    if (!world_.restoreCheckpoint(checkpoint_, &error)) throw std::runtime_error("Cannot reset the world: " + error);
    start_time_ = world_.getWorldTime();
    current_ = CycleResult{};
    pulled_at_ = -1.0;
    return;
  }
  if (time >= kPullStart && time < kPullStart + kPullDuration && world_.getWorldTime() != pulled_at_) {
    pulled_at_ = world_.getWorldTime();
    for (int j = 0; j < kClothRows; ++j) {
      cloth_->setExternalForce(static_cast<std::size_t>(j * kClothColumns), raisim::Vec<3>{-kPullForce / kClothRows, 0.0, -0.2 * kPullForce / kClothRows});
    }
  }
}

inline void Scene::Observe() {
  for (std::size_t k = 0; k < dishes_.size(); ++k) {
    const auto position = dishes_[k]->getPosition();
    if (!position.allFinite()) throw std::runtime_error("Non-finite dish state");
    current_.dish_shift = std::max(current_.dish_shift, std::hypot(position[0] - dish_start_[k][0], position[1] - dish_start_[k][1]));
    current_.dishes_on_table &= position[2] > kTableTop;
  }
  bool off = true;
  for (const auto& p : cloth_->getPositions()) {
    if (!p.e().allFinite()) throw std::runtime_error("Non-finite cloth state");
    off &= p[0] < kTableEdge || p[2] < kTableTop;
  }
  current_.cloth_off_table |= off;
}

}  // namespace raisim_examples::tablecloth

#endif  // RAISIM_EXAMPLES_INCLUDE_TABLECLOTH_PULL_HPP_
