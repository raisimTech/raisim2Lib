// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// A spinning lidar and a probe ray scan a voxel map with hills, a hut, an arch and a tunnel.

#include <cmath>
#include <vector>

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

namespace {

double hill(int x, int y) {
  return 4.0 + 3.0 * std::sin(0.11 * x) * std::cos(0.09 * y) + 2.0 * std::sin(0.05 * (x + y));
}

}  // namespace

int main(int argc, char* argv[]) {
  raisim::World world;
  world.setTimeStep(0.002);

  /// a 30 m x 30 m map of 0.1 m voxels
  auto* voxels = world.addVoxelMap(0.1, {-150, -150, -10}, {300, 300, 80});
  voxels->setName("voxel_terrain");
  voxels->setAppearance("0.62, 0.55, 0.45, 1.0");
  for (int y = -150; y < 150; ++y)
    for (int x = -150; x < 150; ++x)
      voxels->fillBox({x, y, -10}, {x, y, static_cast<int>(hill(x, y))}, true);
  /// a hut with a door
  voxels->fillBox({40, -20, 6}, {80, 20, 40}, true);
  voxels->fillBox({42, -18, 6}, {78, 18, 38}, false);
  voxels->fillBox({38, -5, 6}, {43, 5, 25}, false);
  /// an arch and a tunnel through a hill
  voxels->fillBox({-60, 30, 4}, {-50, 34, 40}, true);
  voxels->fillBox({-20, 30, 4}, {-10, 34, 40}, true);
  voxels->fillBox({-60, 30, 40}, {-10, 34, 46}, true);
  voxels->fillBox({-120, -60, 0}, {-70, -40, 30}, true);
  voxels->fillBox({-125, -54, 2}, {-65, -46, 12}, false);

  raisim::RaisimServer server(&world);
  server.launchServer();
  raisim_examples::warnIfNoClientConnected(server);
  server.setCameraPositionAndLookAt({-12., -12., 10.}, {0., 0., 1.});

  auto* scan = server.addPointCloud("lidar_hits");
  scan->pointSize = 3.f;
  auto* probe = server.addVisualPolyLine("probe_ray");
  probe->color = {1, 0, 0, 1};
  probe->width = 0.03;
  probe->points = {{0, 0, 8}, {0, 0, 0}};
  auto* hitMarker = server.addVisualSphere("probe_hit", 0.12, 1, 0, 0, 1);

  std::vector<raisim::Vec<3>, raisim::AlignedAllocator<raisim::Vec<3>, 32>> hits;
  const raisim::Mat<3, 3> identity = raisim::Mat<3, 3>::getIdentity();
  for (int step = 0;; ++step) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    server.integrateWorldThreadSafe([&]() {
      /// the sensor drives along a circle 1.2 m above the terrain
      const double angle = 0.0008 * step;
      raisim::Vec<3> sensor;
      sensor = {9.0 * std::cos(angle), 9.0 * std::sin(angle), 0.0};
      const auto key = voxels->getKeyAtWorldPosition(sensor);
      sensor[2] = 0.1 * (hill(key[0], key[1]) + 1.0) + 1.2;

      /// 32 x 512 rays per scan
      world.rayTestLidar(identity, sensor, 0.0, 2.0 * M_PI / 512, 512, -0.35, 0.02, 32, 0.0, 30.0,
                         size_t(-10), size_t(-10), raisim::CollisionGroup(-1), hits);
      scan->position.resize(hits.size());
      scan->color.resize(hits.size());
      for (size_t i = 0; i < hits.size(); ++i) {
        scan->position[i] = hits[i] + sensor;
        const double shade = std::min(1.0, hits[i].norm() / 20.0);
        scan->color[i] = {1.0 - shade, 0.4 + 0.6 * shade, 1.0, 1.0};
      }

      /// a single ray sweeps the scene from above
      const Eigen::Vector3d start(0.0, 0.0, 8.0);
      const double sweep = 0.002 * step;
      const Eigen::Vector3d direction =
          Eigen::Vector3d(std::cos(sweep), std::sin(sweep), -1.2 + 0.8 * std::sin(0.3 * sweep)).normalized();
      const auto& rayHits = world.rayTest(start, direction, 40.0);
      if (rayHits.size() > 0) {
        probe->points[1] = rayHits[0].getPosition();
        hitMarker->setPosition(rayHits[0].getPosition());
      }
    });
  }

  server.killServer();
  return 0;
}
