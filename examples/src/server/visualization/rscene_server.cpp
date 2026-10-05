// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.
//
// rscene_server
// -------------
// Serves a RaiSim Engine scene (.rscene) to the rayrai TCP viewer. raisim::World builds
// the physics from the file and RaisimServer streams its bodies. The viewer reads the same
// file for everything else the scene shows: visual-only objects, foliage, lights, the sky
// and the terrain textures.
//
// On this computer the viewer finds the scene and its assets where the server reads them.
// A viewer on another computer (start the server with setBindLoopbackOnly(false)) looks in
// its resource directories and its download cache first, and asks before downloading
// whatever is still missing from this server. setSceneFileSharing(false) turns sharing off.

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "example_resources.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main(int argc, char* argv[]) {
  raisim::World world(exampleRscPath(argv[0], "forest/rayrai_forest.rscene"));

  raisim::RaisimServer server(&world);
  server.setSceneFileSharing(true); // the default; shown for discoverability
  server.launchServer();
  raisim_examples::warnIfNoClientConnected(server);

  for (;;) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    server.integrateWorldThreadSafe();
  }

  server.killServer();
}
