// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
#ifndef RAISIM_EXAMPLES_INCLUDE_SUCTION_PICKING_HPP_
#define RAISIM_EXAMPLES_INCLUDE_SUCTION_PICKING_HPP_

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

// A Kinova arm with a vacuum suction cup stacks four parcels and a can into a
// tower, then puts them back where they were, and starts over. The cup seals
// on the top face of each object; the object hangs from it while the arm
// carries it, and drops off when the vacuum is released.
namespace raisim_examples::suction_picking {

constexpr double kTimeStep = 0.001;
// The arm stands on the table, at the origin of the x-y plane.
constexpr double kTableHeight = 0.32;
// The tower is built here, on the table.
constexpr double kTowerX = 0.4;
constexpr double kTowerY = 0.2;
// The cup comes down this far into the top face, so it seals against it.
constexpr double kPress = 0.004;
// Objects are let go this high above where they come to rest.
constexpr double kDrop = 0.003;
constexpr double kClearance = 0.045;
constexpr double kSpeed = 0.45;
constexpr const char* kTipFrame = "suction_tip_joint";

/// An object to stack: its shape, color and home position on the table.
struct ItemSpec {
  enum class Shape { kBox, kCylinder } shape;
  double x, y, z;  // box sizes, or the cylinder's radius, unused and height [m]
  double mass;
  const char* color;
  double homeX, homeY;
};

// Largest footprint first: the order of the tower from the bottom.
constexpr std::array<ItemSpec, 5> kItems{{
    {ItemSpec::Shape::kBox, 0.11, 0.09, 0.05, 0.30, "0.72,0.55,0.36,1", 0.42, -0.13},
    {ItemSpec::Shape::kBox, 0.10, 0.08, 0.045, 0.25, "0.25,0.42,0.78,1", 0.27, -0.25},
    {ItemSpec::Shape::kBox, 0.09, 0.07, 0.04, 0.20, "0.82,0.26,0.22,1", 0.44, -0.30},
    {ItemSpec::Shape::kBox, 0.08, 0.06, 0.035, 0.15, "0.93,0.76,0.2,1", 0.28, -0.40},
    {ItemSpec::Shape::kCylinder, 0.03, 0.0, 0.07, 0.25, "0.3,0.65,0.35,1", 0.52, -0.02},
}};

/// The arm, the objects and the pick-and-place controller.
class Scene {
 public:
  explicit Scene(const std::string& armUrdf);

  /// Runs the task and the arm controller and integrates one step.
  void step();

  raisim::World& world() { return world_; }
  raisim::ArticulatedSystem* robot() const { return robot_; }
  const std::vector<raisim::SingleBodyObject*>& items() const { return items_; }
  std::size_t cup() const { return cup_; }
  /// Whether the vacuum is on, and whether the cup holds something.
  bool suctionOn() const { return robot_->isGripperActive(cup_); }
  bool holding() const { return robot_->getGripperState(cup_).engaged; }
  /// The number of objects in the tower, and how many towers were built.
  int towerSize() const { return towerSize_; }
  int towers() const { return towers_; }
  /// The number of times the cup failed to seal on an object or lost it before it reached its place.
  int drops() const { return drops_; }
  /// Vertical extent of an item as it lies [m].
  double itemHeight(std::size_t item) const;
  /// Center of the cup lip, where the controller wants it, and where it is [m].
  const Eigen::Vector3d& tipTarget() const { return tipTarget_; }
  Eigen::Vector3d tipPosition() const;

 private:
  enum class Stage { kAbove, kDescend, kSeal, kLift, kCarry, kLower, kRelease, kRetreat, kRetry, kDone };

  struct Move {
    Eigen::Vector3d from = Eigen::Vector3d::Zero();
    Eigen::Vector3d to = Eigen::Vector3d::Zero();
    double duration = 0.0;
  };

  void nextStage();
  void moveTo(const Eigen::Vector3d& goal, double speed = kSpeed);
  std::size_t currentItem() const { return building_ ? order_ : kItems.size() - 1 - order_; }
  /// Whether the cup holds this item.
  bool holds(std::size_t item) const;
  Eigen::Vector3d topOf(std::size_t item) const;
  void track();

  raisim::World world_;
  raisim::ArticulatedSystem* robot_ = nullptr;
  std::vector<raisim::SingleBodyObject*> items_;
  std::size_t cup_ = 0;
  std::size_t tipFrame_ = 0;

  bool building_ = true;
  std::size_t order_ = 0;
  int towerSize_ = 0;
  int towers_ = 0;
  int drops_ = 0;
  Stage stage_ = Stage::kDone;
  double stageTime_ = 0.0;
  double transit_ = 0.0;
  double heldHeight_ = 0.0;
  Eigen::Vector3d place_ = Eigen::Vector3d::Zero();

  Move move_;
  double moveTime_ = 0.0;
  Eigen::Vector3d tipTarget_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d tipOrientation_ = Eigen::Matrix3d::Identity();
  raisim::ArticulatedSystem::IKOptions ik_;
  Eigen::VectorXd command_, commandVelocity_;
};

inline Scene::Scene(const std::string& armUrdf) {
  world_.setTimeStep(kTimeStep);
  world_.setSleepingEnabled(false);
  world_.addGround(0.0, "ground")->setAppearance("checkerboard");
  auto* table = world_.addBox(1.0, 1.1, 0.04, 1.0, "table");
  table->setName("workbench");
  table->setBodyType(raisim::BodyType::STATIC);
  table->setPosition(0.27, -0.05, kTableHeight - 0.02);
  table->setAppearance("0.32,0.36,0.42,1");
  for (double x : {-0.2, 0.74})
    for (double y : {-0.57, 0.47}) {
      auto* leg = world_.addBox(0.03, 0.03, kTableHeight - 0.04, 1.0);
      leg->setBodyType(raisim::BodyType::STATIC);
      leg->setPosition(x, y, 0.5 * (kTableHeight - 0.04));
      leg->setAppearance("0.16,0.18,0.21,1");
    }

  raisim::ArticulatedSystemOption options;
  options.convexifyCollisionMeshes = true;
  robot_ = world_.addArticulatedSystem(armUrdf, "", {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);
  robot_->setName("kinova_suction");
  robot_->setBasePos(raisim::Vec<3>{0.0, 0.0, kTableHeight});
  if (robot_->getDOF() != 6 || robot_->getGrippers().size() != 1)
    throw std::runtime_error("The arm must have six joints and one suction cup");
  cup_ = robot_->getGripperIndex("suction_cup");
  tipFrame_ = robot_->getFrameIdxByName(kTipFrame);

  for (std::size_t i = 0; i < kItems.size(); ++i) {
    const auto& spec = kItems[i];
    raisim::SingleBodyObject* item = nullptr;
    if (spec.shape == ItemSpec::Shape::kBox)
      item = world_.addBox(spec.x, spec.y, spec.z, spec.mass, "parcel");
    else
      item = world_.addCylinder(spec.x, spec.z, spec.mass, "parcel");
    item->setName("item_" + std::to_string(i));
    item->setPosition(spec.homeX, spec.homeY, kTableHeight + 0.5 * spec.z);
    item->setAppearance(spec.color);
    items_.push_back(item);
  }
  world_.setMaterialPairProp("parcel", "table", 0.7, 0.0, 0.001);
  world_.setMaterialPairProp("parcel", "parcel", 0.6, 0.0, 0.001);

  // Joints from the IK demo's home pose; the cup points straight down.
  command_ = Eigen::VectorXd::Zero(6);
  command_ << -0.028, 3.0295, -1.8961, 0.0, 1.3576, -0.028;
  commandVelocity_ = Eigen::VectorXd::Zero(6);
  robot_->setGeneralizedCoordinate(command_);
  robot_->setGeneralizedVelocity(commandVelocity_);
  Eigen::VectorXd pGain(6), dGain(6);
  pGain << 2000.0, 3000.0, 2500.0, 600.0, 600.0, 300.0;
  dGain << 60.0, 80.0, 70.0, 20.0, 20.0, 12.0;
  robot_->setPdGains(pGain, dGain);
  robot_->setPdTarget(command_, commandVelocity_);

  raisim::Mat<3, 3> orientation;
  robot_->getFrameOrientation(tipFrame_, orientation);
  tipOrientation_ = orientation.e();
  tipTarget_ = tipPosition();
  move_ = {tipTarget_, tipTarget_, 0.0};

  ik_.maxIterations = 30;
  ik_.useOrientation = true;
  ik_.applySolution = false;
  ik_.orientationWeight = 0.3;
  nextStage();
}

inline Eigen::Vector3d Scene::tipPosition() const {
  raisim::Vec<3> position;
  robot_->getFramePosition(tipFrame_, position);
  return position.e();
}

inline bool Scene::holds(std::size_t item) const {
  const auto& state = robot_->getGripperState(cup_);
  return state.engaged && state.object == items_[item];
}

inline double Scene::itemHeight(std::size_t item) const {
  const auto& spec = kItems[item];
  const Eigen::Matrix3d rotation = items_[item]->getRotationMatrix();
  if (spec.shape == ItemSpec::Shape::kBox)
    return std::abs(rotation(2, 0)) * spec.x + std::abs(rotation(2, 1)) * spec.y + std::abs(rotation(2, 2)) * spec.z;
  const double upright = std::abs(rotation(2, 2));
  return upright * spec.z + std::sqrt(std::max(0.0, 1.0 - upright * upright)) * 2.0 * spec.x;
}

inline Eigen::Vector3d Scene::topOf(std::size_t item) const {
  return items_[item]->getPosition() + Eigen::Vector3d(0.0, 0.0, 0.5 * itemHeight(item));
}

inline void Scene::moveTo(const Eigen::Vector3d& goal, double speed) {
  const double distance = (goal - tipTarget_).norm();
  move_ = {tipTarget_, goal, std::max(0.35, 1.875 * distance / speed)};
  moveTime_ = 0.0;
}

inline void Scene::nextStage() {
  // The next object: the tower is built bottom up and taken apart top down.
  const std::size_t item = currentItem();
  stageTime_ = 0.0;
  switch (stage_) {
    case Stage::kDone: {
      double towerTop = kTableHeight;
      for (std::size_t i = 0; i < static_cast<std::size_t>(towerSize_); ++i) towerTop += itemHeight(i);
      heldHeight_ = itemHeight(item);
      transit_ = std::max(towerTop, topOf(item).z()) + heldHeight_ + kClearance;
      place_ = building_ ? Eigen::Vector3d(kTowerX, kTowerY, towerTop)
                         : Eigen::Vector3d(kItems[item].homeX, kItems[item].homeY, kTableHeight);
      moveTo(Eigen::Vector3d(topOf(item).x(), topOf(item).y(), transit_));
      stage_ = Stage::kAbove;
      break;
    }
    case Stage::kAbove:
      moveTo(topOf(item) - Eigen::Vector3d(0.0, 0.0, kPress), 0.15);
      robot_->setGripperActive(cup_, true);
      stage_ = Stage::kDescend;
      break;
    case Stage::kDescend:
      stage_ = Stage::kSeal;
      break;
    case Stage::kSeal:
      moveTo(Eigen::Vector3d(tipTarget_.x(), tipTarget_.y(), transit_), 0.25);
      stage_ = Stage::kLift;
      break;
    case Stage::kLift:
      moveTo(Eigen::Vector3d(place_.x(), place_.y(), transit_));
      stage_ = Stage::kCarry;
      break;
    case Stage::kCarry:
      moveTo(place_ + Eigen::Vector3d(0.0, 0.0, heldHeight_ + kDrop), 0.15);
      stage_ = Stage::kLower;
      break;
    case Stage::kLower:
      robot_->setGripperActive(cup_, false);
      stage_ = Stage::kRelease;
      break;
    case Stage::kRelease:
      moveTo(tipTarget_ + Eigen::Vector3d(0.0, 0.0, kClearance), 0.25);
      stage_ = Stage::kRetreat;
      break;
    case Stage::kRetry:
      stage_ = Stage::kDone;
      nextStage();
      break;
    case Stage::kRetreat:
      towerSize_ += building_ ? 1 : -1;
      if (++order_ == kItems.size()) {
        order_ = 0;
        if (building_) towers_++;
        building_ = !building_;
      }
      stage_ = Stage::kDone;
      nextStage();
      break;
  }
}

inline void Scene::step() {
  moveTime_ += kTimeStep;
  stageTime_ += kTimeStep;
  const bool moved = moveTime_ >= move_.duration;
  const bool lost = !holds(currentItem());
  switch (stage_) {
    case Stage::kSeal:
    case Stage::kLift:
    case Stage::kCarry:
    case Stage::kLower:
      if (lost && (stage_ != Stage::kSeal || stageTime_ > 0.25)) {
        // The cup did not seal on the object, or the object tore off: back off
        // and pick it up again where it lies.
        drops_++;
        robot_->setGripperActive(cup_, false);
        moveTo(tipTarget_ + Eigen::Vector3d(0.0, 0.0, kClearance), 0.2);
        stage_ = Stage::kRetry;
        stageTime_ = 0.0;
      } else if (stage_ == Stage::kSeal ? stageTime_ > 0.25 : moved) {
        nextStage();
      }
      break;
    case Stage::kRelease:
      if (stageTime_ > 0.2) nextStage();
      break;
    default:
      if (moved) nextStage();
  }
  track();
  world_.integrate();
}

inline void Scene::track() {
  // Follow the move with a smooth start and stop, through the IK solution of
  // each step, seeded with the last one so the arm stays on one branch.
  const double s = move_.duration > 0.0 ? std::clamp(moveTime_ / move_.duration, 0.0, 1.0) : 1.0;
  const double shape = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s);
  tipTarget_ = move_.from + shape * (move_.to - move_.from);
  ik_.seed.assign(command_.data(), command_.data() + command_.size());
  raisim::Vec<3> position;
  raisim::Mat<3, 3> orientation;
  position.e() = tipTarget_;
  orientation.e() = tipOrientation_;
  const auto result = robot_->solveIK(tipFrame_, position, orientation, ik_);
  const auto& solution = result.success ? result.solution : result.bestSolution;
  if (solution.size() == static_cast<std::size_t>(command_.size())) {
    const Eigen::VectorXd next = Eigen::Map<const Eigen::VectorXd>(solution.data(), command_.size());
    commandVelocity_ = (next - command_) / kTimeStep;
    command_ = next;
  }
  robot_->setPdTarget(command_, commandVelocity_);
}

}  // namespace raisim_examples::suction_picking

#endif  // RAISIM_EXAMPLES_INCLUDE_SUCTION_PICKING_HPP_
