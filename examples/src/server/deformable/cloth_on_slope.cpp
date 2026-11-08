// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// A box on a cloth on each of two ramps (friction 0.5), solved together with
// the cloth (World::setMonolithicDeformables()). Below the friction angle
// (18 degrees) the cloth and the box stay put; above it (30 degrees) they slide
// down together. Every four seconds the world starts over and the example
// prints how far each box moved.

#include <iostream>

#include "raisim/RaisimServer.hpp"
#include "cloth_on_slope.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main() {
  namespace cloth_slope = raisim_examples::cloth_slope;
  cloth_slope::Scene scene;
  raisim::RaisimServer server(&scene.world());
  server.launchServer(8080);
  server.setCameraPositionAndLookAt({1.6, -2.0, 1.2}, {0.0, 0.0, 0.2});
  raisim_examples::warnIfNoClientConnected(server);

  std::size_t reported = 0;
  for (;;) {
    RS_TIMED_LOOP(static_cast<int>(cloth_slope::kTimeStep * 1e6))
    server.integrateWorldThreadSafe([&scene] { scene.Update(); });
    if (scene.completed_cycles() != reported) {
      reported = scene.completed_cycles();
      const auto& result = scene.last_result();
      std::cout << "Cycle " << reported << ": the box on the 18 degree ramp moved " << 1e3 * result.box_travel[0] << " mm, the one on the 30 degree ramp "
                << result.box_travel[1] << " m\n";
    }
  }
}
