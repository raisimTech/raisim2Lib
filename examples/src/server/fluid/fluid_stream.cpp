// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// A stream flows down a channel past a post (World::addFluidVolume() with a current). Crates, logs
// and beach balls float off with it, and bricks on the stony bed stay put until the stream, past about
// 2.1 m/s, slides them along. Drag "Stream speed" in the viewer's Controls window
// (RaisimServer::addSlider()) to change the current; what reaches the end of the channel drops back in
// at its start.

#include <iostream>

#include "fluid_stream.hpp"
#include "raisim/RaisimServer.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main() {
  namespace stream = raisim_examples::fluid_stream;
  stream::DebrisScene scene(0.8);
  raisim::RaisimServer server(&scene.world());
  const size_t speed = server.addSlider("Stream speed [m/s]", 0.0, stream::kMaxSpeed, 0.8);
  server.launchServer(8080);
  server.setCameraPositionAndLookAt({1.0, -9.0, 5.0}, {4.0, 0.0, 0.0});
  raisim_examples::warnIfNoClientConnected(server);

  size_t reported = 0;
  for (;;) {
    RS_TIMED_LOOP(static_cast<int>(stream::kTimeStep * 1e6))
    const double streamSpeed = server.getSlider(speed);
    server.integrateWorldThreadSafe([&] { scene.update(streamSpeed); });
    if (scene.recycled() != reported) {
      reported = scene.recycled();
      std::cout << reported << " bodies carried to the end of the channel by a stream of " << streamSpeed << " m/s\n";
    }
  }
}
