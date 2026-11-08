// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "example_resources.hpp"
#include "raisim/RaisimServer.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main(int argc, char* argv[]) {
  /// create raisim world
  raisim::World world;
  world.setTimeStep(0.001);

  /// create objects
  auto movingGround = world.addBox(10, 10, 1, 10, "gnd");
  movingGround->setBodyType(raisim::BodyType::KINEMATIC);  // kinematic objects have infinite mass
  movingGround->setPosition(0, 0, -0.5);
  movingGround->setAppearance("red");
  raisim::ArticulatedSystemOption options;
  options.convexifyCollisionMeshes = true;
  auto anymalB = world.addArticulatedSystem(exampleRscPath(argv[0], "anymal/urdf/anymal.urdf"), "", {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);

  /// anymalC joint PD controller
  Eigen::VectorXd jointNominalConfig(anymalB->getGeneralizedCoordinateDim()), jointVelocityTarget(anymalB->getDOF());
  jointNominalConfig << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
  jointVelocityTarget.setZero();

  Eigen::VectorXd jointPgain(anymalB->getDOF()), jointDgain(anymalB->getDOF());
  jointPgain.tail(12).setConstant(100.0);
  jointDgain.tail(12).setConstant(1.0);

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
  server.focusOn(anymalB);

  raisim_examples::warnIfNoClientConnected(server);
  for (int i = 0; i < 20000; i++) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    movingGround->setLinearVelocity({0, 0, 3. * sin(double(i) / 3000. * M_PI)});
    server.integrateWorldThreadSafe();
    //    world.integrate();
  }

  server.killServer();
  return 0;
}
