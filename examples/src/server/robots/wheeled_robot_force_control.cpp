// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "example_resources.hpp"
#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main(int argc, char* argv[]) {
  raisim::World world;
  world.setTimeStep(0.002);

  /// create raisim objects
  auto ground = world.addGround();
  ground->setAppearance("0.55,0.55,0.55,1.0");

  raisim::ArticulatedSystemOption options;
  options.convexifyCollisionMeshes = true;
  auto robot = world.addArticulatedSystem(exampleRscPath(argv[0], "megabot/smb.urdf"), "", {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);

  robot->setName("smb");
  Eigen::VectorXd gc(robot->getGeneralizedCoordinateDim()), gv(robot->getDOF());
  gc.setZero();
  gv.setZero();
  //  raisim::Vec<4> quat; quat = {0, 0.0499792, 0, 0.9987503}; quat/= quat.norm();
  gc.segment<7>(0) << 0, 0, 0.197, 1, 0, 0, 0;

  robot->setGeneralizedCoordinate(gc);
  robot->setGeneralizedVelocity(gv);
  robot->setGeneralizedForce({0, 0, 0, 0, 0, 0, 6, 6, 6, 6});
  robot->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::SEMI_IMPLICIT);
  robot->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);

  /// launch raisim server
  raisim::RaisimServer server(&world);
  server.launchServer();
  raisim_examples::warnIfNoClientConnected(server);
  server.focusOn(robot);

  for (int i = 0; i < 20000000; i++) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    server.integrateWorldThreadSafe();
  }

  server.killServer();
  return 0;
}
