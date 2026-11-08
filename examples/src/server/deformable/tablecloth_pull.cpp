// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// The tablecloth trick: three dishes stand on a cloth on a table, and a sharp
// pull on the cloth's edge whips it out from under them. The cloth is solved
// together with the dishes and the table (World::setMonolithicDeformables()).
// Every three seconds the world starts over and the example prints how far the
// dishes moved.

#include <iostream>

#include "raisim/RaisimServer.hpp"
#include "rayrai_tcp_viewer_hint.hpp"
#include "tablecloth_pull.hpp"

int main() {
  namespace tablecloth = raisim_examples::tablecloth;
  tablecloth::Scene scene;
  raisim::RaisimServer server(&scene.world());
  server.launchServer(8080);
  server.setCameraPositionAndLookAt({-1.6, -1.4, 1.5}, {-0.1, 0.0, 0.75});
  raisim_examples::warnIfNoClientConnected(server);

  std::size_t reported = 0;
  for (;;) {
    RS_TIMED_LOOP(static_cast<int>(tablecloth::kTimeStep * 1e6))
    server.integrateWorldThreadSafe([&scene] { scene.Update(); });
    if (scene.completed_cycles() != reported) {
      reported = scene.completed_cycles();
      const auto& result = scene.last_result();
      std::cout << "Cycle " << reported << ": the cloth " << (result.cloth_off_table ? "came off" : "stayed on") << " the table, the dishes moved up to "
                << 1e3 * result.dish_shift << " mm and " << (result.dishes_on_table ? "stayed on" : "fell off") << " the table\n";
    }
  }
}
