#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "raisim/World.hpp"
#include "raisim/object/terrain/HeightMap.hpp"
#include "forest_tree_index.hpp"

namespace forest {
constexpr int samples = 161;
constexpr double extent = 80;
inline double trail(double y) { return 3.5 * std::sin(y * .10); }
inline double elevation(double x, double y) {
  const double hills = 3.4 * std::sin(x * .065) * std::cos(y * .055)
    + 1.8 * std::sin(y * .12 + x * .035) + .38 * std::sin(x * .32) * std::cos(y * .27);
  const double gully = 1.8 * std::exp(-std::pow((x - 13 - 4 * std::sin(y * .08)) / 5, 2));
  return hills - gully;
}
inline raisim::HeightMap* addTerrain(raisim::World& world) {
  std::vector<double> heights;
  std::vector<raisim::ColorRGB> colors;
  heights.reserve(samples * samples);
  colors.reserve(samples * samples);
  for (int y = 0; y < samples; ++y) for (int x = 0; x < samples; ++x) {
    const double px = -extent / 2 + extent * x / (samples - 1);
    const double py = -extent / 2 + extent * y / (samples - 1);
    heights.push_back(elevation(px, py));
    colors.push_back({105,119,78});
  }
  auto* terrain = world.addHeightMap(samples, samples, extent, extent, 0, 0, heights);
  terrain->setName("Forest terrain");
  terrain->setColor(colors);
  return terrain;
}
struct Plant { double x, y, z, scale, yaw; int type; };
inline std::vector<Plant> scatter(const raisim::HeightMap& terrain) {
  std::mt19937 rng(42);
  // Explicit conversion keeps the scatter identical across standard libraries.
  auto unit = [&] { return double(rng()) / 4294967296.0; };
  std::vector<Plant> plants;
  plants.reserve(2 * 880 + 128 + 3 * 18000 + 4 * 1800);
  TreeProximityIndex trees;
  for (int type = 0; type < 10; ++type) {
    const bool tree = type < 3;
    const int count = type < 2 ? 880 : type == 2 ? 128 : type < 6 ? 18000 : 1800;
    for (int n = 0, attempts = 0; n < count && attempts < count * 100; ++attempts) {
      // Low vegetation covers the full terrain, including the open tree corridor.
      const double span = tree ? 74 : extent;
      double x = (unit() - .5) * span, y = (unit() - .5) * span;
      if (tree && (std::abs(x - trail(y)) < 3.0 || std::hypot(x, y + 9) < 4)) continue;
      // Let crowns overlap while keeping neighboring trunks separate.
      if (tree && trees.anyWithin(x, y, .8)) continue;
      const double z = terrain.getHeight(x, y);
      if (tree) {
        const double slope = std::hypot(terrain.getHeight(x+.25,y)-terrain.getHeight(x-.25,y),
                                       terrain.getHeight(x,y+.25)-terrain.getHeight(x,y-.25)) / .5;
        if (slope > .65) continue;
      }
      const double base = type < 2 ? 4.0 : type == 2 ? 1.5 : type == 3 ? 3 : type == 4 ? 3.5 : 1.0;
      plants.push_back({x,y,z,base*(.8+.4*unit()),unit()*6.283185307179586,type});
      if (tree) trees.add(x, y);
      ++n;
    }
  }
  return plants;
}
// Six normalized rock meshes, each contained in a horizontal radius of one metre.
inline std::vector<Plant> scatterRocks(const raisim::HeightMap& terrain,
                                     const std::vector<Plant>& plants) {
  std::mt19937 rng(93);
  auto unit = [&] { return double(rng()) / 4294967296.0; };
  std::vector<Plant> rocks;
  rocks.reserve(180);
  TreeProximityIndex trees;
  for (const auto& p : plants) if (p.type < 3) trees.add(p.x, p.y);
  for (int attempts = 0; rocks.size() < 180 && attempts < 18000; ++attempts) {
    const double y = (unit()-.5)*72;
    const double x = attempts%2 ? (unit()-.5)*72
      : trail(y)+(unit()<.5 ? -1 : 1)*(2.8+unit()*2.2);
    const double scale = .35 + .85*unit();
    if (std::abs(x-trail(y)) < 2.1+1.2*scale || std::hypot(x,y+9) < 4+scale) continue;
    if (trees.anyWithin(x, y, scale+.25)) continue;
    if (std::any_of(rocks.begin(),rocks.end(),[&](const Plant& p) {
          return std::hypot(x-p.x,y-p.y) < scale+p.scale;
        })) continue;
    // Embed the rounded underside using terrain samples around its contact patch.
    double low = terrain.getHeight(x,y), high = low;
    for (int i=0;i<8;++i) {
      const double a = i*6.283185307179586/8;
      const double z = terrain.getHeight(x+.35*scale*std::cos(a),y+.35*scale*std::sin(a));
      low=std::min(low,z); high=std::max(high,z);
    }
    if (high-low > .35*scale) continue;
    rocks.push_back({x,y,low-.06*scale,scale,unit()*6.283185307179586,int(rocks.size()%6)});
  }
  return rocks;
}
enum class DemoShape { Box, Sphere, Capsule };
enum class DemoColor { Blue, Coral, Gold, Cyan, Violet };

struct DemoPrimitive {
  std::string name;
  DemoShape shape;
  DemoColor color;
  double x, y, z;
  std::array<double, 3> size{1, 1, 1};
  double radius = .3;
  double height = .6;
  double mass = 1;
  int targetTree = -1;
};

inline const char* demoColorName(DemoColor color) {
  switch (color) {
    case DemoColor::Blue: return "blue";
    case DemoColor::Coral: return "coral";
    case DemoColor::Gold: return "gold";
    case DemoColor::Cyan: return "cyan";
    case DemoColor::Violet: return "violet";
  }
  throw std::logic_error("Unknown forest demo color");
}

inline std::array<double, 4> demoColorRgba(DemoColor color) {
  switch (color) {
    case DemoColor::Blue: return {.15, .35, .7, 1};
    case DemoColor::Coral: return {.95, .25, .12, 1};
    case DemoColor::Gold: return {.95, .72, .12, 1};
    case DemoColor::Cyan: return {.08, .72, .9, 1};
    case DemoColor::Violet: return {.65, .35, .9, 1};
  }
  throw std::logic_error("Unknown forest demo color");
}

inline std::vector<DemoPrimitive> demoPrimitives(
    const raisim::HeightMap& terrain, const std::vector<Plant>& plants) {
  std::vector<DemoPrimitive> result;
  result.reserve(100);
  // Keep the original physics clearing and stack visible shapes along the
  // camera's first stretch of trail. The upper tiers fall onto lower tiers.
  for (int i = 0; i < 6; ++i) {
    const double x = (i % 3 - 1) * 1.15, y = -9 + (i / 3) * 1.4;
    result.push_back({"Falling crate " + std::to_string(i), DemoShape::Box,
      i % 2 ? DemoColor::Coral : DemoColor::Blue,
      x, y, terrain.getHeight(x, y) + 1.4, {.65, .65, .65}, .3, .6, 3});
    result.push_back({"Rolling ball " + std::to_string(i), DemoShape::Sphere,
      DemoColor::Gold, x, y + 3, terrain.getHeight(x, y + 3) + 2});
  }
  constexpr std::array colors{DemoColor::Coral, DemoColor::Cyan,
    DemoColor::Gold, DemoColor::Violet, DemoColor::Blue};
  for (int row = 0; row < 6; ++row) {
    const double y = -17.0 + row * 3.0;
    for (int lane = 0; lane < 4; ++lane) {
      const double x = trail(y) + (lane - 1.5) * .85;
      const double ground = terrain.getHeight(x, y);
      for (int tier = 0; tier < 3; ++tier) {
        const int index = (row * 4 + lane) * 3 + tier;
        const int shape = (row + lane + tier) % 3;
        result.push_back({"Trail primitive " + std::to_string(index),
          shape == 0 ? DemoShape::Box :
            shape == 1 ? DemoShape::Sphere : DemoShape::Capsule,
          colors[(row + lane + tier) % colors.size()],
          x, y, ground + 1.8 + tier * 1.3 + row * .12,
          {.55, .55, .55}, .28, .65, 1.5});
      }
    }
  }
  std::vector<std::size_t> nearby;
  for (std::size_t i = 0; i < plants.size(); ++i) {
    const auto& tree = plants[i];
    if (tree.type >= 3) continue;
    const double sideways = std::abs(tree.x - trail(tree.y));
    if (tree.type < 3 && tree.y > -20 && tree.y < 2 &&
        sideways >= 3 && sideways <= 6.5)
      nearby.push_back(i);
  }
  std::sort(nearby.begin(), nearby.end(), [&](std::size_t a, std::size_t b) {
    const auto score = [&](const Plant& tree) {
      return std::pow(tree.y + 12, 2) +
        std::pow(std::abs(tree.x - trail(tree.y)) - 3.5, 2);
    };
    const double left = score(plants[a]), right = score(plants[b]);
    return left == right ? a < b : left < right;
  });
  if (nearby.size() < 16)
    throw std::runtime_error("Forest has too few nearby trees for collision drops");
  for (int i = 0; i < 16; ++i) {
    const auto& tree = plants[nearby[i]];
    const double towardTrail = tree.x < trail(tree.y) ? .2 : -.2;
    result.push_back({"Tree drop " + std::to_string(i), DemoShape::Sphere,
      i % 2 ? DemoColor::Gold : DemoColor::Violet,
      tree.x + towardTrail, tree.y,
      tree.z + tree.scale * (tree.type == 2 ? 3.2 : 1.1) + 1.0,
      {1, 1, 1}, .42, .6, 1.5, static_cast<int>(nearby[i])});
  }
  return result;
}

inline void addObjects(raisim::World& world, const raisim::HeightMap& terrain,
                       const std::vector<Plant>& plants) {
  world.setTimeStep(.002);
  for (const auto& spec : demoPrimitives(terrain, plants)) {
    raisim::SingleBodyObject* body = nullptr;
    switch (spec.shape) {
      case DemoShape::Box:
        body = world.addBox(spec.size[0], spec.size[1], spec.size[2], spec.mass);
        break;
      case DemoShape::Sphere:
        body = world.addSphere(spec.radius, spec.mass);
        break;
      case DemoShape::Capsule:
        body = world.addCapsule(spec.radius, spec.height, spec.mass);
        break;
    }
    if (!body) throw std::logic_error("Unknown forest demo shape");
    body->setPosition(spec.x, spec.y, spec.z);
    const auto color = demoColorRgba(spec.color);
    body->setAppearance(std::to_string(color[0]) + "," +
      std::to_string(color[1]) + "," + std::to_string(color[2]) + ",1");
    body->setName(spec.name);
  }
}
} // namespace forest
