// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "raisim/RaisimServer.hpp"
#include "rayrai_tcp_viewer_hint.hpp"
#include "example_resources.hpp"

int main(int argc, char* argv[]) {
  raisim::World::setActivationKey(exampleRscPath(argv[0], "activation.raisim"));

  /// create raisim world
  raisim::World world;
  world.setTimeStep(0.001);

  /// create objects
  auto ground = world.addGround(0, "gnd");
  ground->setAppearance("checkerboard");
  auto anymalB = world.addArticulatedSystem(exampleRscPath(argv[0], "anymal/urdf/anymal.urdf"));
  auto anymalC = world.addArticulatedSystem(exampleRscPath(argv[0], "anymal_c/urdf/anymal.urdf"));

  /// anymalC joint PD controller
  Eigen::VectorXd jointNominalConfig(anymalC->getGeneralizedCoordinateDim()), jointVelocityTarget(anymalC->getDOF());
  jointNominalConfig << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
  jointVelocityTarget.setZero();

  Eigen::VectorXd jointPgain(anymalC->getDOF()), jointDgain(anymalC->getDOF());
  jointPgain.tail(12).setConstant(100.0);
  jointDgain.tail(12).setConstant(1.0);

  anymalC->setGeneralizedCoordinate(jointNominalConfig);
  anymalC->setGeneralizedForce(Eigen::VectorXd::Zero(anymalC->getDOF()));
  anymalC->setPdGains(jointPgain, jointDgain);
  anymalC->setPdTarget(jointNominalConfig, jointVelocityTarget);
  anymalC->setName("anymalC");

  jointNominalConfig[1] = 1.0;
  anymalB->setGeneralizedCoordinate(jointNominalConfig);
  anymalB->setGeneralizedForce(Eigen::VectorXd::Zero(anymalB->getDOF()));
  anymalB->setPdGains(jointPgain, jointDgain);
  anymalB->setPdTarget(jointNominalConfig, jointVelocityTarget);
  anymalB->setName("anymalB");

  /// friction example. uncomment it to see the effect
//  anymalB->getCollisionBody("LF_FOOT/0").setMaterial("LF_FOOT");
//  world.setMaterialPairProp("gnd", "LF_FOOT", 0.01, 0, 0);

  /// launch raisim server
  raisim::RaisimServer server(&world);
  server.launchServer();
  server.focusOn(anymalC);
  raisim_examples::warnIfNoClientConnected(server);

  for (int i=0; i<200000000; i++) {
    RS_TIMED_LOOP(int(world.getTimeStep()*1e6))
    server.integrateWorldThreadSafe();
  }

  server.killServer();
}
