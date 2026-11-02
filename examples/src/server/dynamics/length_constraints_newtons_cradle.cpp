// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "example_resources.hpp"
#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main(int argc, char** argv) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);

  /// create raisim world
  raisim::World world;
  world.setTimeStep(0.001);

  /// create raisim objects
  auto ground = world.addGround();

  // simulator
  world.setMaterialPairProp("steel", "steel", 0.1, 1.0, 0.0);

  auto pin1 = world.addSphere(0.1, 0.8);
  pin1->setAppearance("1,0,0,0.3");
  pin1->setPosition(0.0, 0.0, 3.0);
  pin1->setBodyType(raisim::BodyType::STATIC);

  auto pin2 = world.addSphere(0.1, 0.8);
  pin2->setAppearance("0,1,0,0.3");
  pin2->setPosition(0.3, 0.0, 3.0);
  pin2->setBodyType(raisim::BodyType::STATIC);

  auto pin3 = world.addSphere(0.1, 0.8);
  pin3->setAppearance("0,0,1,0.3");
  pin3->setPosition(0.6, 0.0, 3.0);
  pin3->setBodyType(raisim::BodyType::STATIC);

  auto pin4 = world.addSphere(0.1, 0.8);
  pin4->setAppearance("1,0,0,0.3");
  pin4->setPosition(0.9, 0.0, 3.0);
  pin4->setBodyType(raisim::BodyType::STATIC);

  auto pin5 = world.addSphere(0.1, 0.8);
  pin5->setPosition(0.9, 0.0, 6.0);
  pin5->setBodyType(raisim::BodyType::STATIC);

  auto pin6 = world.addSphere(0.1, 0.8);
  pin6->setPosition(-3., 0.0, 7.0);
  pin6->setBodyType(raisim::BodyType::STATIC);

  auto pin7 = world.addSphere(0.1, 0.8);
  pin7->setPosition(-4., 0.0, 7.0);
  pin7->setBodyType(raisim::BodyType::STATIC);

  raisim::ArticulatedSystemOption options;
  options.convexifyCollisionMeshes = true;
  auto anymalC = world.addArticulatedSystem(exampleRscPath(argv[0], "anymal_c/urdf/anymal.urdf"), "", {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);
  auto anymalB = world.addArticulatedSystem(exampleRscPath(argv[0], "anymal/urdf/anymal.urdf"), "", {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);

  /// anymalC joint PD controller
  Eigen::VectorXd jointNominalConfig(anymalC->getGeneralizedCoordinateDim()), jointVelocityTarget(anymalC->getDOF());
  jointNominalConfig << -3, 0, 4.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
  jointVelocityTarget.setZero();

  Eigen::VectorXd jointPgain(anymalC->getDOF()), jointDgain(anymalC->getDOF());
  jointPgain.tail(12).setConstant(100.0);
  jointDgain.tail(12).setConstant(1.0);

  anymalC->setGeneralizedCoordinate(jointNominalConfig);
  anymalC->setGeneralizedForce(Eigen::VectorXd::Zero(anymalC->getDOF()));
  anymalC->setPdGains(jointPgain, jointDgain);
  anymalC->setPdTarget(jointNominalConfig, jointVelocityTarget);
  anymalC->setName("anymalC");

  jointNominalConfig[0] = -4;
  anymalB->setGeneralizedCoordinate(jointNominalConfig);
  anymalB->setGeneralizedForce(Eigen::VectorXd::Zero(anymalB->getDOF()));
  anymalB->setPdGains(jointPgain, jointDgain);
  anymalB->setPdTarget(jointNominalConfig, jointVelocityTarget);
  anymalB->setName("anymalB");

  auto ball1 = world.addSphere(0.1498, 0.8, "steel");
  ball1->setPosition(0, 0.0, 1.0);

  auto ball2 = world.addSphere(0.1499, 0.8, "steel");
  ball2->setPosition(0.3, 0.0, 1.0);

  auto ball3 = world.addSphere(0.1499, 0.8, "steel");
  ball3->setPosition(0.6, 0.0, 1.0);

  auto ball4 = world.addSphere(0.1499, 0.8, "steel");
  ball4->setPosition(2.9, 0.0, 3.0);

  auto box = world.addBox(.1, .1, .1, 1);
  box->setPosition(0.9, 0.0, 4.2);

  using Path = raisim::Tendon::PathElement;
  raisim::Tendon::Properties cable;
  cable.upperLimit = 2.0;
  cable.width = 0.02;
  cable.color = {1., 1., 1., 1.};
  auto redCable = cable;
  redCable.width = 0.05;
  redCable.color = {1., 0., 0., 1.};
  world.addSpatialTendon("cradle_1", {Path::via({pin1}), Path::via({ball1})}, redCable);
  world.addSpatialTendon("cradle_2", {Path::via({pin2}), Path::via({ball2})}, cable);
  world.addSpatialTendon("cradle_3", {Path::via({pin3}), Path::via({ball3})}, cable);
  world.addSpatialTendon("cradle_4", {Path::via({pin4}), Path::via({ball4})}, cable);

  raisim::Tendon::Properties spring;
  spring.springLower = spring.springUpper = 2.0;
  spring.stiffness = 200.;
  spring.width = 0.02;
  world.addSpatialTendon("box_spring", {Path::via({pin5}), Path::via({box})}, spring);
  spring.stiffness = 1000.;
  world.addSpatialTendon("robot_spring", {Path::via({pin6}), Path::via({anymalC})}, spring);

  auto* actuator = world.addSpatialTendon("robot_lift", {Path::via({pin7}), Path::via({anymalB})});
  actuator->setTension(310.);

  /// launch raisim server
  raisim::RaisimServer server(&world);
  server.launchServer(8080);

  raisim_examples::warnIfNoClientConnected(server, 8080);
  world.exportToXml(binaryPath.getDirectory(), "exportedWorld.xml");

  for (int i = 0; i < 5000000; i++) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    if (server.isConnected()) {
      server.integrateWorldThreadSafe();
    }

    if (i == 5000) world.removeTendon(actuator);
  }

  server.killServer();
  return 0;
}