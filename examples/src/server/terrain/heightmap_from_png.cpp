// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include <raisim/RaisimServer.hpp>
#include <raisim/World.hpp>

#include "example_resources.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main(int argc, char* argv[]) {
  /// create raisim world
  raisim::World world;
  world.setTimeStep(0.001);

  /// create objects
  auto heightMap = world.addHeightMap(exampleRscPath(argv[0], "xmlScripts/heightMaps/zurichHeightMap.png"), 0, 0, 500, 500, 0.005, -10);
  raisim::ArticulatedSystemOption options;
  options.convexifyCollisionMeshes = true;
  auto anymal = world.addArticulatedSystem(exampleRscPath(argv[0], "anymal/urdf/anymal.urdf"), "", {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);
  anymal->setGeneralizedCoordinate({0, 0, 10.8, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8});
  anymal->setName("anymal");
  heightMap->setAppearance("soil1");

  /// launch raisim server
  raisim::RaisimServer server(&world);
  server.launchServer();

  raisim_examples::warnIfNoClientConnected(server);
  for (int i = 0; i < 50000; i++) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    server.integrateWorldThreadSafe();
  }

  server.killServer();
  return 0;
}