// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// Heavy loads on cloth, solved together with the bodies
// (World::setMonolithicDeformables()): a 10 kg ball rests on a cloth on the
// ground, a 5 kg box on two stacked cloths, and a 5 kg blanket on a 1 kg box.
// Each load rests on what carries the cloth instead of pressing the particles
// into it. Every five seconds the world starts over and the example prints how
// far each load pressed into what carries it.

#include <iostream>

#include "raisim/RaisimServer.hpp"
#include "heavy_objects_on_cloth.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main() {
  namespace heavy_cloth = raisim_examples::heavy_cloth;
  heavy_cloth::Scene scene;
  raisim::RaisimServer server(&scene.world());
  server.launchServer(8080);
  server.setCameraPositionAndLookAt({0.0, -2.2, 1.1}, {0.0, 0.0, 0.1});
  raisim_examples::warnIfNoClientConnected(server);

  std::size_t reported = 0;
  for (;;) {
    RS_TIMED_LOOP(static_cast<int>(heavy_cloth::kTimeStep * 1e6))
    server.integrateWorldThreadSafe([&scene] { scene.Update(); });
    if (scene.completed_cycles() != reported) {
      reported = scene.completed_cycles();
      const auto& result = scene.last_result();
      std::cout << "Cycle " << reported << ": cloth under the ball in the ground " << 1e3 * result.ball_cloth_in_ground << " mm, upper cloth in the lower one "
                << 1e3 * result.layers_overlap << " mm, box under the blanket in the ground " << 1e3 * result.blanket_box_in_ground << " mm\n";
    }
  }
}
