// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include <iostream>

#include "example_resources.hpp"
#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#if WIN32
#include <timeapi.h>
#endif

int main(int argc, char* argv[]) {
  raisim::World::setActivationKey(exampleRscPath(argv[0], "activation.raisim"));

  if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    std::cout << "Usage: " << argv[0] << " [XML_WORLD]\n"
              << "Loads objects/SingleBodies.xml by default. Relative paths are also searched "
                 "under rsc/xmlScripts."
              << std::endl;
    return 0;
  }

  const std::string xmlScript = argc > 1 ? argv[1] : "objects/SingleBodies.xml";
  raisim::Path xmlPath(xmlScript);
  if (!xmlPath.fileExists()) {
    xmlPath = exampleRscPath(argv[0], "xmlScripts/" + xmlScript);
  }
  if (!xmlPath.fileExists()) {
    std::cerr << "Could not find XML world: " << xmlScript << std::endl;
    return 1;
  }

  std::cout << "Loading RaiSim world: " << xmlPath.getString() << std::endl;
  raisim::World world(xmlPath.getString());
  raisim::RaisimServer server(&world);
  server.launchServer();
  for (int i = 0; i < 10000000; i++) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    server.integrateWorldThreadSafe();
  }

  server.killServer();
  return 0;
}
