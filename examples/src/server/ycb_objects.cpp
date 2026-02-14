// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

#include <string>
#include <vector>

int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);
  raisim::RaiSimMsg::setFatalCallback([]() { throw; });

  raisim::World world;
  world.setTimeStep(0.001);

  world.addGround();

  const std::string sep = raisim::Path::separator();
  std::string basePath = binaryPath.getDirectory().getString() + sep + "rsc" + sep + "ycb" + sep;
  raisim::Path::replaceAntiSeparatorWithSeparator(basePath);

  const std::vector<std::string> filenames = {"002_master_chef_can.urdf", "007_tuna_fish_can.urdf",
    "012_strawberry.urdf", "013_apple.urdf"};

  raisim::ArticulatedSystem* focus = nullptr;
  for (size_t i = 0; i < filenames.size(); ++i) {
    auto obj = world.addArticulatedSystem(basePath + filenames[i]);
    obj->setBasePos({-1.0, 0.3 * double(i), 0.2});
    obj->setName("ycb_" + std::to_string(i));
    if (!focus)
      focus = obj;
  }

  raisim::RaisimServer server(&world);
  server.launchServer();
  raisim_examples::warnIfNoClientConnected(server);
  if (focus)
    server.focusOn(focus);

  for (int i = 0; i < 2000000; i++) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    server.integrateWorldThreadSafe();
  }

  server.killServer();
  return 0;
}
