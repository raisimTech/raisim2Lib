// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "example_resources.hpp"

int main(int argc, char* argv[]) {
  raisim::World::setActivationKey(exampleRscPath(argv[0], "activation.raisim"));

  std::vector<raisim::World::ParameterContainer> params;
  params.push_back({"spawn_sphere", "true"});
  params.push_back({"spawn_box", "false"});
  params.push_back({"sphere_count", "30"});
  params.push_back({"sphere_height_offset", "3"});
  params.push_back({"laikago_start_x", "-2"});
  params.push_back({"laikago_start_y", "0"});
  params.push_back({"floor_height", "-1"});

  raisim::World world(exampleRscPath(argv[0], "xmlScripts/templatedWorld/templatedWorld.xml"), params);
  raisim::RaisimServer server(&world);
  server.launchServer();
  for (int i=0; i<10000000; i++) {
    RS_TIMED_LOOP(int(world.getTimeStep()*1e6))
    server.integrateWorldThreadSafe();
  }

  server.killServer();
}