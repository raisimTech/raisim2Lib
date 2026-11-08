// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// Bodies of different densities fall into a pool (World::addFluidVolume()). The water holds each up
// with the weight of the water it displaces, at the center of that volume: the ice block floats
// with 8 % above the surface, the half-dense wooden cube turns an edge up, the raft and the log
// float flat, the hollow ball rides high, and the steel ball and the brick sink. Every fifteen
// seconds the world starts over and the example prints how each came to rest.

#include <iostream>

#include "fluid_pool.hpp"
#include "raisim/RaisimServer.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main() {
  namespace pool = raisim_examples::fluid_pool;
  pool::Scene scene;
  raisim::RaisimServer server(&scene.world());
  server.launchServer(8080);
  server.setCameraPositionAndLookAt({0.0, -4.5, 2.6}, {0.0, 0.0, 0.5});
  raisim_examples::warnIfNoClientConnected(server);

  std::size_t reported = 0;
  for (;;) {
    RS_TIMED_LOOP(static_cast<int>(pool::kTimeStep * 1e6))
    server.integrateWorldThreadSafe([&scene] { scene.update(); });
    if (scene.completed_cycles() != reported) {
      reported = scene.completed_cycles();
      const auto& result = scene.last_result();
      std::cout << "Cycle " << reported << ": " << 100.0 * result.ice_above << " % of the ice above the water, the cube "
                << "tilted by " << result.cube_tilt_deg << " degrees, the raft " << 100.0 * result.raft_draft
                << " cm deep, the steel ball " << 100.0 * result.steel_height << " cm over the floor\n";
    }
  }
}
