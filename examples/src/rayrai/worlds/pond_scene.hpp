#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

#include "raisim/World.hpp"
#include "raisim/object/terrain/HeightMap.hpp"
#include "rayrai/Water.hpp"

namespace pond {

constexpr double kExtent = 90;
constexpr int kSamples = 181;
constexpr double kWaterLevel = 0;
constexpr double kCenterX = 0, kCenterY = 10;
constexpr double kYaw = .22;
constexpr double kSemiMajor = 15, kSemiMinor = 9;
constexpr double kMaxDepth = 1.7;
constexpr double kWaterExtentScale = 1.3;
constexpr double kSunAzimuthDegrees = 244, kSunElevationDegrees = 28;
constexpr double kCameraYawDegrees = 82, kCameraPitchDegrees = -8, kCameraFovDegrees = 60;
constexpr double kPi = 3.14159265358979323846;

inline double smoothstep(double lo, double hi, double x) {
  const double t = std::clamp((x - lo) / (hi - lo), 0.0, 1.0);
  return t * t * (3 - 2 * t);
}

// Pond frame: u along the long axis, v along the short one, both scaled so the
// mean shoreline is the unit ellipse.
inline std::array<double, 2> pondFrame(double x, double y) {
  const double dx = x - kCenterX, dy = y - kCenterY;
  const double c = std::cos(kYaw), s = std::sin(kYaw);
  return {(c * dx + s * dy) / kSemiMajor, (-s * dx + c * dy) / kSemiMinor};
}

inline std::array<double, 2> worldFromPond(double u, double v) {
  const double c = std::cos(kYaw), s = std::sin(kYaw);
  const double a = u * kSemiMajor, b = v * kSemiMinor;
  return {kCenterX + c * a - s * b, kCenterY + s * a + c * b};
}

inline double shoreRadius(double angle) {
  return 1 + .07 * std::sin(3 * angle + .5) + .04 * std::sin(5 * angle + 2.1) + .03 * std::sin(7 * angle + .3);
}

// 1 on the shoreline, below 1 in the water.
inline double pondRadius(double x, double y) {
  const auto [u, v] = pondFrame(x, y);
  return std::hypot(u, v) / shoreRadius(std::atan2(v, u));
}

inline double hills(double x, double y) {
  const double rolling = .9 * std::sin(x * .085 + .4) * std::cos(y * .07) + .5 * std::sin(y * .11 + x * .04) +
                         .15 * std::sin(x * .31) * std::cos(y * .27);
  return 1.55 + rolling + 3 * smoothstep(24, 44, y);
}

inline double elevation(double x, double y) {
  const double r = pondRadius(x, y);
  if (r < 1) {
    const double t = 1 - r * r;
    return kWaterLevel - kMaxDepth * t * (.55 + .45 * t);
  }
  return kWaterLevel + .85 * (1 - std::exp(-3 * (r - 1))) + smoothstep(1.15, 2.4, r) * hills(x, y);
}

inline raisim::HeightMap* addTerrain(raisim::World& world) {
  std::vector<double> heights;
  std::vector<raisim::ColorRGB> colors;
  heights.reserve(kSamples * kSamples);
  colors.reserve(kSamples * kSamples);
  for (int j = 0; j < kSamples; ++j)
    for (int i = 0; i < kSamples; ++i) {
      const double x = -kExtent / 2 + kExtent * i / (kSamples - 1);
      const double y = -kExtent / 2 + kExtent * j / (kSamples - 1);
      const double z = elevation(x, y);
      heights.push_back(z);
      const double grass = smoothstep(kWaterLevel + .05, kWaterLevel + .45, z);
      colors.push_back({static_cast<unsigned char>(118 - 22 * grass), static_cast<unsigned char>(104 + 18 * grass),
                        static_cast<unsigned char>(76 - 4 * grass)});
    }
  auto* terrain = world.addHeightMap(kSamples, kSamples, kExtent, kExtent, 0, 0, heights);
  terrain->setName("Pond terrain");
  terrain->setColor(colors);
  return terrain;
}

inline raisin::WaterBody waterBody() {
  raisin::WaterBody water = raisin::defaultWaterBody(raisin::WaterPreset::CalmLake);
  water.shape = raisin::WaterBodyShape::Rectangle;
  water.center = {kCenterX, kCenterY, kWaterLevel};
  water.yawRadians = static_cast<float>(kYaw);
  water.halfExtents = {static_cast<float>(kSemiMajor * kWaterExtentScale), static_cast<float>(kSemiMinor * kWaterExtentScale)};
  water.depth = static_cast<float>(kMaxDepth + .5);
  raisin::WaterWaveSpectrum wind;
  wind.waveCount = 12;
  wind.seed = 7;
  wind.windAngleRadians = 1.2f;
  wind.directionSpreadRadians = .9f;
  wind.minWavelength = .3f;
  wind.maxWavelength = 4;
  wind.minAmplitude = .002f;
  wind.maxAmplitude = .012f;
  wind.longWaveSteepness = .25f;
  wind.shortWaveSteepness = .35f;
  water.waves = raisin::generateGerstnerWaves(wind);
  water.absorption = {.55f, .30f, .95f};
  water.scattering = {.02f, .035f, .015f};
  water.roughness = .02f;
  water.rippleStrength = .22f;
  water.rippleScale = 1.4f;
  water.rippleDirection = {std::cos(1.2f), std::sin(1.2f)};
  water.shorelineFoamDepth = 0;
  water.crestFoamStrength = 0;
  water.causticStrength = .45f;
  water.causticScale = 1.1f;
  return water;
}

// The camera stands on the south bank, looking north across the water toward the low sun.
inline std::array<double, 3> cameraPosition() {
  const double x = .5, y = -4.5;
  return {x, y, elevation(x, y) + 1.7};
}

inline std::array<double, 2> towardSun() {
  const double az = kSunAzimuthDegrees * kPi / 180;
  return {-std::cos(az), -std::sin(az)};
}

// Trees stay out of the view from the camera toward the sun, so the sun and its
// reflection are not hidden behind them.
inline bool inSunCorridor(double x, double y) {
  const auto camera = cameraPosition();
  const auto sun = towardSun();
  const double dx = x - camera[0], dy = y - camera[1];
  const double along = dx * sun[0] + dy * sun[1];
  const double across = std::abs(dx * sun[1] - dy * sun[0]);
  return along > -3 && across < 2.5 + .12 * std::max(along, 0.0);
}

enum PlantType { kTree, kBermuda, kGrass, kTallGrass, kFern, kDandelion, kNettle, kPeriwinkle, kPlantTypes };

struct Placement {
  double x, y, z, scale, yaw;
  int type;
};

inline bool onLand(double x, double y, double margin) {
  return std::abs(x) < kExtent / 2 - 1 && std::abs(y) < kExtent / 2 - 1 && elevation(x, y) > kWaterLevel + margin;
}

inline std::vector<Placement> scatterPlants() {
  std::mt19937 rng(11);
  const auto unit = [&] { return static_cast<double>(rng()) / 4294967296.0; };
  const auto camera = cameraPosition();
  std::vector<Placement> plants;
  constexpr std::array<int, kPlantTypes> counts{70, 5000, 5000, 3500, 900, 500, 500, 700};
  constexpr std::array<double, kPlantTypes> baseScale{2.2, 3, 3.5, 1, 1, 1, 1, 1};
  for (int type = 0; type < kPlantTypes; ++type) {
    const bool tree = type == kTree;
    for (int placed = 0, attempts = 0; placed < counts[type] && attempts < counts[type] * 200; ++attempts) {
      const double x = (unit() - .5) * (kExtent - 2), y = (unit() - .5) * (kExtent - 2);
      const double r = pondRadius(x, y);
      if (!onLand(x, y, tree ? .4 : .03)) continue;
      if (tree && (r < 1.4 || inSunCorridor(x, y) || std::hypot(x - camera[0], y - camera[1]) < 7)) continue;
      // Grass gathers along the banks and thins out away from the water.
      if (!tree && unit() > .25 + .75 * (1 - smoothstep(1.1, 3.2, r))) continue;
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

// Rocks along the waterline, half in the water, and a few on the banks.
inline std::vector<Placement> scatterRocks() {
  std::mt19937 rng(23);
  const auto unit = [&] { return static_cast<double>(rng()) / 4294967296.0; };
  std::vector<Placement> rocks;
  for (int attempts = 0; rocks.size() < 36 && attempts < 20000; ++attempts) {
    const bool shore = rocks.size() < 22;
    const double angle = 2 * kPi * unit();
    const double radius = shore ? .97 + .1 * unit() : 1.3 + 1.6 * unit();
    const auto p = worldFromPond(radius * shoreRadius(angle) * std::cos(angle), radius * shoreRadius(angle) * std::sin(angle));
    const double scale = shore ? .35 + .5 * unit() : .3 + .7 * unit();
    if (std::abs(p[0]) > kExtent / 2 - 2 || std::abs(p[1]) > kExtent / 2 - 2 || inSunCorridor(p[0], p[1])) continue;
    if (std::any_of(rocks.begin(), rocks.end(), [&](const Placement& q) { return std::hypot(q.x - p[0], q.y - p[1]) < scale + q.scale + .3; }))
      continue;
    rocks.push_back({p[0], p[1], elevation(p[0], p[1]) - .1 * scale, scale, 2 * kPi * unit(), static_cast<int>(rocks.size() % 6)});
  }
  return rocks;
}

// A wooden jetty from the south-west bank into the pond: deck boards along it on two beams,
// standing on posts that reach the pond floor. Box sizes are along the jetty, across it and up.
struct JettyPart {
  enum Kind { kBoard, kBeam, kPost } kind;
  std::array<double, 3> size;
  std::array<double, 3> position;
  double heading;
};

inline std::vector<JettyPart> jettyParts() {
  const auto start = worldFromPond(-.32, -1.2), end = worldFromPond(-.32, -.58);
  const double dx = end[0] - start[0], dy = end[1] - start[1];
  const double length = std::hypot(dx, dy), heading = std::atan2(dy, dx);
  const double ux = dx / length, uy = dy / length;
  const double deck = kWaterLevel + .45;
  const auto at = [&](double along, double side) {
    return std::array<double, 2>{start[0] + ux * along - uy * side, start[1] + uy * along + ux * side};
  };
  std::vector<JettyPart> parts;
  for (int board = 0; board < 7; ++board) {
    const auto p = at(length / 2, (board - 3) * .23);
    parts.push_back({JettyPart::kBoard, {length, .2, .06}, {p[0], p[1], deck + .005 * (board % 2)}, heading});
  }
  for (int side : {-1, 1}) {
    const auto p = at(length / 2, side * .72);
    parts.push_back({JettyPart::kBeam, {length, .12, .14}, {p[0], p[1], deck - .1}, heading});
  }
  for (int i = 0; i * 2.2 < length; ++i)
    for (int side : {-1, 1}) {
      const auto p = at(.6 + i * 2.2, side * .72);
      const double bottom = std::min(elevation(p[0], p[1]), kWaterLevel) - .3;
      parts.push_back({JettyPart::kPost, {.07, .07, deck - bottom}, {p[0], p[1], (deck + bottom) / 2}, heading});
    }
  return parts;
}

}  // namespace pond
