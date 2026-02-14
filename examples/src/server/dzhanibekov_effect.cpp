// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main () {
  raisim::World world;
  world.setTimeStep(0.001);
  raisim::RaisimServer server(&world);

  world.setGravity(raisim::Vec<3>{0,0,0});
  auto box = world.addBox(0.5,1,3,5);
  box->setPosition(0, 0, 0);
  box->setAngularVelocity(raisim::Vec<3>{0.0001,5,0});
  server.launchServer();
  raisim_examples::warnIfNoClientConnected(server);
  server.focusOn(box);

  for (int i=0; i<100000; i++) {
    RS_TIMED_LOOP(1e6 * world.getTimeStep())
    server.integrateWorldThreadSafe();
  }

  server.killServer();

  return 0;
}
