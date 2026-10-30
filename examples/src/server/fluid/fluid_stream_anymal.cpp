// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// ANYmal stands across a stream 0.55 m deep (World::addFluidVolume() with a current), the water
// pushing on its side and holding part of its weight up, its joints holding the standing pose. Drag
// "Stream speed" in the viewer's Controls window (RaisimServer::addSlider()): a gentle stream leaves it
// standing; past about 1.2 m/s the water tips it over and carries it off downstream. Once it reaches
// the end of the channel, or has lain fallen for four seconds, it stands up at the start again.

#include <iostream>

#include "example_resources.hpp"
#include "fluid_stream.hpp"
#include "raisim/RaisimServer.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main(int, char* argv[]) {
  namespace stream = raisim_examples::fluid_stream;
  stream::RobotScene scene(exampleRscPath(argv[0], "anymal/urdf/anymal.urdf"), 0.5);
  raisim::RaisimServer server(&scene.world());
  const size_t speed = server.addSlider("Stream speed [m/s]", 0.0, stream::kMaxSpeed, 0.5);
  server.launchServer(8080);
  server.setCameraPositionAndLookAt({2.0, -6.0, 2.6}, {2.5, 0.0, 0.3});
  raisim_examples::warnIfNoClientConnected(server);

  size_t reported = 0;
  for (;;) {
    RS_TIMED_LOOP(static_cast<int>(stream::kTimeStep * 1e6))
    const double streamSpeed = server.getSlider(speed);
    server.integrateWorldThreadSafe([&] { scene.update(streamSpeed); });
    if (scene.falls() != reported) {
      reported = scene.falls();
      std::cout << "A stream of " << streamSpeed << " m/s knocked ANYmal over and carried it off (" << reported
                << " so far); it stands up at the start again\n";
    }
  }
}
