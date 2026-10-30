// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include <algorithm>
#include <filesystem>
#include <iostream>

#include "example_resources.hpp"
#include "fluid_rov_visuals.hpp"
#include "raisim/PathUtils.hpp"
#include "raisim/RaisimServer.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main(int, char* argv[]) {
  namespace rov = raisim_examples::fluid_rov;
  const auto assets = std::filesystem::absolute(raisim::pathFromUtf8(exampleRscPath(argv[0], "underwater_rov"))).lexically_normal();
  rov::Scene scene(raisim::pathUtf8(assets / "underwater_rov.rscene"));
  raisim::RaisimServer server(&scene.world());
  rov::VehicleVisuals visuals(scene, server, raisim::pathUtf8(assets / "rov.glb"));
  server.launchServer(8080);
  server.setCameraPositionAndLookAt({1.8, -2.7, 3.8}, {-0.15, 0.1, 2.5});
  raisim_examples::warnIfNoClientConnected(server);

  double worst = 0.0, error = 0.0, time = 0.0;
  int reported = 0;
  for (;;) {
    RS_TIMED_LOOP(static_cast<int>(rov::kTimeStep * 1e6))
    server.integrateWorldThreadSafe([&] {
      scene.update();
      visuals.update();
      error = scene.positionError();
      time = scene.world().getWorldTime();
    });
    worst = std::max(worst, error);
    const int period = static_cast<int>(time / rov::kCurrentPeriod);
    if (period != reported) {
      reported = period;
      std::cout << "The current turned: it pushed the vehicle up to " << 100.0 * worst << " cm off its station, "
                << "which it held to " << 100.0 * error << " cm\n";
      worst = 0.0;
    }
  }
}
