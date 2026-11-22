// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// Primitives and meshes fall onto a voxel map with stairs, walls, rubble and a pit; every few
// seconds a new crater is carved into the map and the bodies on it fall in.

#include <cmath>
#include <string>

#include "example_resources.hpp"
#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main(int argc, char* argv[]) {
  raisim::World world;
  world.setTimeStep(0.002);

  /// a 12 m x 12 m map of 5 cm voxels on 3 m of ground; the map ends below its domain, so craters
  /// must not reach through the ground
  auto* voxels = world.addVoxelMap(0.05, {-120, -120, -60}, {240, 240, 120});
  voxels->setName("voxel_playground");
  voxels->setAppearance("0.55, 0.52, 0.48, 1.0");
  voxels->fillBox({-120, -120, -60}, {119, 119, -1}, true);
  /// a low rim around the edge
  voxels->fillBox({-120, -120, 0}, {119, -118, 5}, true);
  voxels->fillBox({-120, 117, 0}, {119, 119, 5}, true);
  voxels->fillBox({-120, -120, 0}, {-118, 119, 5}, true);
  voxels->fillBox({117, -120, 0}, {119, 119, 5}, true);
  /// stairs down into a pit
  voxels->fillBox({-40, -40, -16}, {40, 40, -1}, false);
  for (int s = 0; s < 8; ++s) voxels->fillBox({-40 + 4 * s, -40, -16}, {-37 + 4 * s, -20, -1 - 2 * s}, true);
  /// a wall with windows, rubble and pillars
  voxels->fillBox({60, -80, 0}, {63, 80, 30}, true);
  for (int w = -60; w < 60; w += 30) voxels->fillBox({60, w, 10}, {63, w + 10, 20}, false);
  for (int i = 0; i < 80; ++i) {
    const int x = (i * 37) % 180 - 90, y = (i * 53) % 180 - 90;
    if (std::abs(x) < 45 && std::abs(y) < 45) continue;
    voxels->fillBox({x, y, 0}, {x + 1 + i % 5, y + 1 + (i * 3) % 4, i % 6}, true);
  }
  for (int p = 0; p < 4; ++p) voxels->fillBox({-90 + 40 * p, 70, 0}, {-88 + 40 * p, 72, 40}, true);

  const std::string monkey = exampleRscPath(argv[0], "monkey/monkey.obj");
  const std::string uShape = exampleRscPath(argv[0], "rayrai/coacd/coacd_u_shape.obj");
  const std::string arch = exampleRscPath(argv[0], "rayrai/coacd/coacd_arch.obj");

  raisim::RaisimServer server(&world);
  server.launchServer();
  raisim_examples::warnIfNoClientConnected(server);
  server.setCameraPositionAndLookAt({-7., -7., 6.}, {0., 0., -0.3});

  int dropped = 0;
  for (int step = 0;; ++step) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    server.integrateWorldThreadSafe([&]() {
      /// a new body every 0.3 s, until 120 are in the scene
      if (step % 150 == 0 && dropped < 120) {
        raisim::SingleBodyObject* body = nullptr;
        switch (dropped % 6) {
          case 0: body = world.addSphere(0.12, 1.0); break;
          case 1: body = world.addBox(0.3, 0.2, 0.15, 1.0); break;
          case 2: body = world.addCapsule(0.07, 0.25, 1.0); break;
          case 3: body = world.addCylinder(0.1, 0.15, 1.0); break;
          case 4: body = world.addMesh(monkey, 1.0, 0.25, "", raisim::MeshCollisionMode::CONVEX_HULL); break;
          default:
            body = world.addMesh(dropped % 12 == 5 ? uShape : arch, 1.0, 0.3, "",
                                 raisim::MeshCollisionMode::ORIGINAL_MESH);
            break;
        }
        const double angle = 0.7 * dropped;
        body->setPosition(2.5 * std::cos(angle), 2.5 * std::sin(angle), 3.0);
        body->setOrientation(std::cos(0.5 * angle), std::sin(0.5 * angle), 0.0, 0.0);
        body->setAppearance(std::to_string(0.3 + 0.12 * (dropped % 5)) + ", 0.45, " +
                            std::to_string(0.9 - 0.1 * (dropped % 4)) + ", 1.0");
        ++dropped;
      }
      /// carve a new crater every 8 s; the bodies on it fall in
      if (step % 4000 == 0) {
        raisim::Vec<3> crater;
        crater = {3.0 * std::cos(0.0005 * step), 3.0 * std::sin(0.0005 * step), 0.0};
        voxels->fillSphere(crater, 1.0, false);
      }
    });
  }

  server.killServer();
  return 0;
}
