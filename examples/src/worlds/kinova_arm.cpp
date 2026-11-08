// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "example_resources.hpp"
#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main(int argc, char* argv[]) {
  raisim::World::setActivationKey(exampleRscPath(argv[0], "activation.raisim"));

  /// create raisim world
  raisim::World world;
  world.setTimeStep(0.001);

  /// create objects
  world.addGround();
  raisim::ArticulatedSystemOption options;
  options.convexifyCollisionMeshes = true;
  auto kinova = world.addArticulatedSystem(exampleRscPath(argv[0], "kinova/urdf/kinova.urdf"), "", {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);

  /// kinova joint PD controller
  Eigen::VectorXd jointNominalConfig(kinova->getGeneralizedCoordinateDim()), jointVelocityTarget(kinova->getDOF());
  jointNominalConfig << 0.0, 2.76, -1.57, 0.0, 2.0, 0.0;
  jointVelocityTarget.setZero();

  Eigen::VectorXd jointPgain(kinova->getDOF()), jointDgain(kinova->getDOF());
  jointPgain << 40.0, 40.0, 40.0, 15.0, 15.0, 15.0;
  jointDgain << 1.0, 1.0, 1.0, 0.5, 0.5, 0.5;

  kinova->setGeneralizedCoordinate(jointNominalConfig);
  kinova->setGeneralizedForce(Eigen::VectorXd::Zero(kinova->getDOF()));
  kinova->setPdGains(jointPgain, jointDgain);
  kinova->setPdTarget(jointNominalConfig, jointVelocityTarget);
  kinova->setName("kinova");

  /// launch raisim server
  raisim::RaisimServer server(&world);
  server.launchServer();
  server.focusOn(kinova);
  raisim_examples::warnIfNoClientConnected(server);

  for (int i = 0; i < 2000000; i++) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    server.integrateWorldThreadSafe();
  }

  std::cout << "mass " << kinova->getMassMatrix()[0] << std::endl;

  server.killServer();
  return 0;
}
