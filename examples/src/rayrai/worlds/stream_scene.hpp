#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "raisim/World.hpp"
#include "raisim/object/terrain/HeightMap.hpp"
#include "rayrai/Water.hpp"

namespace stream {

constexpr double kPi = 3.14159265358979323846;
constexpr double kExtentX = 40, kExtentY = 28;
constexpr int kSamplesX = 161, kSamplesY = 113;
// The creek runs down the slope along +x into a pool and out through the east edge.
constexpr double kSlope = .075;
constexpr double kPoolX = 12, kPoolRadius = 3.2, kPoolDepth = .8;
constexpr double kWaterHalfX = 19, kWaterHalfY = 6;
constexpr double kSourceX = -17.5;
constexpr double kSunAzimuthDegrees = 168, kSunElevationDegrees = 27;
constexpr double kWaterDensity = 1000, kGravity = 9.81;

inline double smoothstep(double lo, double hi, double x) {
  const double t = std::clamp((x - lo) / (hi - lo), 0.0, 1.0);
  return t * t * (3 - 2 * t);
}

// Middle of the creek across the slope.
inline double creekY(double x) {
  return 1.8 * std::sin(.17 * x + .6) + .7 * std::sin(.43 * x + 1.3);
}

// Bed along the middle of the creek: a steady slope with two short drops.
inline double creekBed(double x) {
  return -kSlope * x - .3 * smoothstep(-7.4, -6.6, x) - .25 * smoothstep(2.1, 2.9, x);
}

inline double poolDistance(double x, double y) {
  return std::hypot(x - kPoolX, y - creekY(kPoolX)) / kPoolRadius;
}

inline double elevation(double x, double y) {
  const double d = std::abs(y - creekY(x));
  const double channel = -.3 * std::exp(-(d * d) / (.75 * .75));
  const double banks = .25 * smoothstep(.6, 2.0, d) + .65 * smoothstep(1.5, 6.0, d);
  const double hills = .35 * std::sin(.21 * x + .7 * y) * std::cos(.17 * y) * smoothstep(3, 10, d);
  const double r = poolDistance(x, y);
  const double pool = r < 1 ? -kPoolDepth * (1 - r * r) * (.6 + .4 * (1 - r * r)) : 0;
  return creekBed(x) + channel + banks + hills + pool;
}

inline raisim::HeightMap* addTerrain(raisim::World& world) {
  std::vector<double> heights;
  heights.reserve(kSamplesX * kSamplesY);
  for (int j = 0; j < kSamplesY; ++j)
    for (int i = 0; i < kSamplesX; ++i)
      heights.push_back(elevation(-kExtentX / 2 + kExtentX * i / (kSamplesX - 1), -kExtentY / 2 + kExtentY * j / (kSamplesY - 1)));
  auto* terrain = world.addHeightMap(kSamplesX, kSamplesY, kExtentX, kExtentY, 0, 0, heights);
  terrain->setName("Creek terrain");
  return terrain;
}

// The simulated water: a rectangle over the creek that starts dry and fills from a spring at the top.
inline raisin::WaterBody waterBody() {
  raisin::WaterBody water = raisin::defaultWaterBody(raisin::WaterPreset::CalmLake);
  water.shape = raisin::WaterBodyShape::Rectangle;
  water.center = {0, 0, -3.2};
  water.halfExtents = {static_cast<float>(kWaterHalfX), static_cast<float>(kWaterHalfY)};
  water.depth = .5f;
  water.edgeFadeDistance = 0;
  water.waves.clear();
  water.absorption = {.45f, .16f, .22f};
  water.scattering = {.03f, .05f, .04f};
  water.roughness = .02f;
  water.rippleStrength = .45f;
  water.rippleScale = .7f;
  water.shorelineFoamDepth = 0;
  water.crestFoamStrength = 0;
  water.causticStrength = .6f;
  water.causticScale = .6f;
  water.flow.enabled = true;
  water.flow.cellSize = .08f;
  water.flow.bedCeiling = 6;
  water.flow.manningRoughness = .035f;
  water.flow.foamStrength = .7f;
  raisin::WaterFlowSource spring;
  spring.position = {kSourceX, creekY(kSourceX)};
  spring.radius = .5f;
  spring.flowRate = .18f;
  spring.velocity = {.8f, 0};
  water.flow.sources.push_back(spring);
  return water;
}

// The camera stands on the bank above the pool and looks up the creek toward the sun.
inline std::array<double, 3> cameraPosition() {
  const double x = 17, y = creekY(17) - 2.8;
  return {x, y, elevation(x, y) + 2.1};
}

inline std::array<double, 3> cameraTarget() {
  return {4, creekY(4), creekBed(4)};
}

enum PlantType { kTree, kBermuda, kGrass, kTallGrass, kFern, kDandelion, kNettle, kPeriwinkle, kPlantTypes };

struct Placement {
  double x, y, z, scale, yaw;
  int type;
};

// Distance from the middle of the creek, the pool counted as part of it.
inline double creekDistance(double x, double y) {
  return std::min(std::abs(y - creekY(x)), std::max(0.0, (poolDistance(x, y) - 1) * kPoolRadius));
}

inline std::vector<Placement> scatterPlants() {
  std::mt19937 rng(17);
  const auto unit = [&] { return static_cast<double>(rng()) / 4294967296.0; };
  const auto camera = cameraPosition();
  std::vector<Placement> plants;
  constexpr std::array<int, kPlantTypes> counts{60, 4500, 4500, 3000, 800, 400, 400, 600};
  constexpr std::array<double, kPlantTypes> baseScale{2.2, 3, 3.5, 1, 1, 1, 1, 1};
  for (int type = 0; type < kPlantTypes; ++type) {
    const bool tree = type == kTree;
    for (int placed = 0, attempts = 0; placed < counts[type] && attempts < counts[type] * 200; ++attempts) {
      const double x = (unit() - .5) * (kExtentX - 2), y = (unit() - .5) * (kExtentY - 2);
      const double d = creekDistance(x, y);
      if (d < (tree ? 3.5 : 1.2)) continue;
      if (tree && (std::hypot(x - camera[0], y - camera[1]) < 9 || (x < camera[0] && std::abs(y - creekY(x)) < 5))) continue;
      // Grass gathers along the banks and thins out away from the creek.
      if (!tree && unit() > .25 + .75 * (1 - smoothstep(1.5, 6, d))) continue;
      if (tree && std::any_of(plants.begin(), plants.end(), [&](const Placement& p) {
            return p.type == kTree && std::hypot(p.x - x, p.y - y) < 3;
          }))
        continue;
      plants.push_back({x, y, elevation(x, y), baseScale[type] * (.8 + .4 * unit()), 2 * kPi * unit(), type});
      ++placed;
    }
  }
  return plants;
}

// Rocks in the riffles and at the drops, where the water breaks around them (the first
// kCreekRocks), and some on the banks.
constexpr int kCreekRocks = 10;

inline std::vector<Placement> scatterRocks() {
  std::mt19937 rng(29);
  const auto unit = [&] { return static_cast<double>(rng()) / 4294967296.0; };
  std::vector<Placement> rocks;
  for (double x : {-14.0, -11.5, -7.3, -6.8, -3.0, -.5, 2.4, 2.9, 6.0, 8.2}) {
    const double y = creekY(x) + (unit() - .5) * .9;
    const double scale = .35 + .3 * unit();
    rocks.push_back({x, y, elevation(x, y) - .05, scale, 2 * kPi * unit(), static_cast<int>(rocks.size() % 6)});
  }
  for (int attempts = 0; rocks.size() < 30 && attempts < 20000; ++attempts) {
    const double x = (unit() - .5) * (kExtentX - 4), y = creekY(x) + (unit() < .5 ? -1 : 1) * (1.3 + 1.2 * unit());
    const double scale = .3 + .5 * unit();
    if (std::abs(y) > kExtentY / 2 - 2 || poolDistance(x, y) < 1.1) continue;
    if (std::any_of(rocks.begin(), rocks.end(), [&](const Placement& q) { return std::hypot(q.x - x, q.y - y) < scale + q.scale + .3; }))
      continue;
    rocks.push_back({x, y, elevation(x, y) - .1 * scale, scale, 2 * kPi * unit(), static_cast<int>(rocks.size() % 6)});
  }
  return rocks;
}

// The rocks in the creek stop what drifts down it: a hidden collider of each one's convex hull
// (@p forest: the directory with the rock_moss_N models).
inline void addCreekRockColliders(raisim::World& world, const std::filesystem::path& forest) {
  const auto rocks = scatterRocks();
  for (int i = 0; i < kCreekRocks; ++i) {
    const auto& p = rocks[static_cast<size_t>(i)];
    const auto file = forest / ("rock_moss_" + std::to_string(p.type)) / "model.gltf";
    auto* rock = world.addMesh(file.string(), 1, p.scale, "", raisim::MeshCollisionMode::CONVEX_HULL);
    rock->setBodyType(raisim::BodyType::STATIC);
    rock->setPosition(p.x, p.y, p.z);
    rock->setOrientation(std::cos(p.yaw * .5), 0, 0, std::sin(p.yaw * .5));
    rock->setName("creek rock " + std::to_string(i));
    rock->setAppearance("hidden");
  }
}

// Force of the water on a floating ball of radius r and the density given: buoyancy of the volume
// below the surface and the drag of the water flowing past it.
inline std::array<double, 3> floatingForce(const std::array<double, 3>& position, const std::array<double, 3>& velocity,
                                           double radius, const raisin::WaterFlowSample& water, double dragCoefficient = .8) {
  if (!water.wet) return {0, 0, 0};
  const double submerged = std::clamp(water.surfaceHeight - (position[2] - radius), 0.0, 2 * radius);
  if (submerged <= 0) return {0, 0, 0};
  const double volume = kPi * submerged * submerged * (3 * radius - submerged) / 3;
  const double area = kPi * radius * radius * submerged / (2 * radius);
  const std::array<double, 3> relative{velocity[0] - water.velocity.x, velocity[1] - water.velocity.y, velocity[2]};
  const double speed = std::sqrt(relative[0] * relative[0] + relative[1] * relative[1] + relative[2] * relative[2]);
  const double drag = .5 * kWaterDensity * dragCoefficient * area * speed;
  return {-drag * relative[0], -drag * relative[1], kWaterDensity * kGravity * volume - drag * relative[2]};
}

}  // namespace stream
