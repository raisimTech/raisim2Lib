// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "drone_flight.hpp"
#include "example_resources.hpp"
#include "raisim/RaisimServer.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main(int argc, char* argv[]) {
  namespace flight = raisim_examples::drone_flight;
  raisim::World::setActivationKey(exampleRscPath(argv[0], "activation.raisim"));

  flight::Scene scene;
  raisim::RaisimServer server(&scene.world());
  auto* pad = server.addVisualCylinder("pad", 0.45, 0.004, 0.95, 0.75, 0.1, 1.0);
  pad->setPosition(0.0, 0.0, 0.002);
  auto* path = server.addVisualPolyLine("figure_eight");
  path->color = {0.2, 0.55, 0.95, 0.6};
  for (int i = 0; i <= 200; ++i) {
    Eigen::Vector3d point, tangent, curvature;
    flight::eight(2.0 * flight::kPi * i / 200.0, point, tangent, curvature);
    path->points.push_back({point.x(), point.y(), point.z()});
  }
  server.launchServer();
  server.focusOn(scene.drone());
  raisim_examples::warnIfNoClientConnected(server);

  for (int i = 0; i < 10000000; ++i) {
    RS_TIMED_LOOP(int(flight::kTimeStep * 1e6))
    server.lockVisualizationServerMutex();
    scene.step();
    server.unlockVisualizationServerMutex();
  }

  server.killServer();
  return 0;
}
