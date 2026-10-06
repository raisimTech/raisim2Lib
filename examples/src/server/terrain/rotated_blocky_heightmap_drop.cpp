// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"
#include "example_resources.hpp"

int main(int argc, char* argv[]) {
  raisim::World world;
  world.setTimeStep(0.002);
  world.setERP(0.0, 0.0);

  /// the height map of blocky_heightmap_drop: 250x250 samples over 20m x 20m
  /// (0.08m sample spacing). Every 5x5 patch of samples shares one height,
  /// giving 50x50 flat tiles of 0.4m x 0.4m. Tile heights are sampled uniformly
  /// from -0.15m to 0.15m.
  constexpr size_t samples = 250;
  constexpr double mapSize = 20.0;
  constexpr size_t blockSize = 5;
  constexpr size_t blocks = samples / blockSize;
  constexpr double heightRange = 0.15;

  std::mt19937 generator(2024);
  std::uniform_real_distribution<double> heightDistribution(-heightRange,
                                                            heightRange);

  std::vector<double> blockHeights(blocks * blocks);
  for (auto& height : blockHeights) height = heightDistribution(generator);

  std::vector<double> height(samples * samples);
  for (size_t yIndex = 0; yIndex < samples; ++yIndex) {
    for (size_t xIndex = 0; xIndex < samples; ++xIndex) {
      const size_t block =
          (yIndex / blockSize) * blocks + (xIndex / blockSize);
      height[yIndex * samples + xIndex] = blockHeights[block];
    }
  }

  auto* heightMap = world.addHeightMap(samples, samples, mapSize, mapSize, 0.0,
                                       0.0, height);
  heightMap->setName("rotated_blocky_terrain");
  heightMap->setAppearance("soil1");

  /// tilt the whole map by 15 degrees about the world x axis, around its center
  /// (here the world origin): its +y half rises and its -y half drops by up to
  /// 2.6m. Collision, ray tests and rendering follow the rotation.
  constexpr double pi = 3.14159265358979323846;
  constexpr double tiltDegrees = 15.0;
  const Eigen::Matrix3d tilt =
      Eigen::AngleAxisd(tiltDegrees * pi / 180.0, Eigen::Vector3d::UnitX())
          .toRotationMatrix();
  heightMap->setOrientation(tilt);

  /// spheres and cylinders roll down the slope: a ground plane below the lowest
  /// edge catches them
  world.addGround(-4.0);

  /// monkey mesh used for one fifth of the dropped bodies
  const std::string monkeyPath = exampleRscPath(argv[0], "monkey/monkey.obj");

  /// the 900 bodies of blocky_heightmap_drop (30x30 grid): box, sphere, capsule,
  /// cylinder, and monkey mesh, 180 of each, laid out over the map as there and
  /// tilted with it. The compact, centered layout keeps every body fully over
  /// the map with at least 0.8m clearance.
  constexpr int gridRows = 30;
  constexpr int gridColumns = 30;
  constexpr double spacing = 0.6;
  constexpr double dropHeight = 3.0;
  constexpr double maxBodyRadius = 0.5;
  static_assert(gridRows * gridColumns == 900);
  static_assert(0.5 * (gridColumns - 1) * spacing + maxBodyRadius <
                0.5 * mapSize);
  static_assert(0.5 * (gridRows - 1) * spacing + maxBodyRadius <
                0.5 * mapSize);

  int meshCount = 0;
  for (int row = 0; row < gridRows; ++row) {
    for (int column = 0; column < gridColumns; ++column) {
      raisim::SingleBodyObject* object = nullptr;
      const int shape = (row + 2 * column) % 5;

      switch (shape) {
        case 0:
          object = world.addBox(0.35, 0.35, 0.35, 1.0);
          object->setAppearance("blue");
          break;
        case 1:
          object = world.addSphere(0.2, 1.0);
          object->setAppearance("red");
          break;
        case 2:
          object = world.addCapsule(0.15, 0.35, 1.0);
          object->setAppearance("green");
          break;
        case 3:
          object = world.addCylinder(0.2, 0.35, 1.0);
          object->setAppearance("0.8, 0.65, 0.2, 1.0");
          break;
        default: {
          /// a convex hull keeps 180 mesh bodies cheap to load and to collide
          const double scale = 0.13 + 0.01 * (meshCount % 4);
          object = world.addMesh(monkeyPath, 1.0, scale, "",
                                 raisim::MeshCollisionMode::CONVEX_HULL);
          object->setAppearance("0.75, 0.45, 0.25, 1.0");

          /// tilt each monkey differently so that they do not land identically
          const double angle = 0.37 * meshCount;
          object->setOrientation(std::cos(0.5 * angle),
                                 std::sin(0.5 * angle) * std::cos(0.7 * meshCount),
                                 std::sin(0.5 * angle) * std::sin(0.7 * meshCount),
                                 0.0);
          ++meshCount;
          break;
        }
      }

      /// the body's place over the map, in the map's own frame: getHeight()
      /// ignores the orientation by design, so it returns the tile height at
      /// (x, y) of the unrotated map. The tilt carries that surface point into
      /// the world, and the body starts dropHeight above it, turned with the map.
      const double x = spacing * (column - 0.5 * (gridColumns - 1));
      const double y = spacing * (row - 0.5 * (gridRows - 1));
      const Eigen::Vector3d surface =
          tilt * Eigen::Vector3d(x, y, heightMap->getHeight(x, y));
      const double z =
          surface.z() + dropHeight + 0.05 * ((7 * row + 3 * column) % 5);

      object->setName((shape == 4 ? "monkey_" : "primitive_") +
                      std::to_string(row) + "_" + std::to_string(column));
      object->setPosition(surface.x(), surface.y(), z);
      object->setOrientation(Eigen::Matrix3d(tilt * object->getRotationMatrix()));
    }
  }

  raisim::RaisimServer server(&world);
  server.launchServer();

  raisim_examples::warnIfNoClientConnected(server);
  while (true) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    server.integrateWorldThreadSafe();
  }

  server.killServer();
}
