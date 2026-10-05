//
// Created by jemin on 2022-07-21.
//

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"
#include "example_resources.hpp"

int main(int argc, char **argv) {
  const int loopN = 200000000;

  raisim::World world;
  raisim::RaisimServer server(&world);

  auto checkerBoard = world.addGround(0.0, "glass");

  Eigen::VectorXd jointConfig(19), jointVelocityTarget(18);
  Eigen::VectorXd jointState(18), jointVel(18), jointPgain(18), jointDgain(18);

  jointPgain.setZero();
  jointPgain.tail(12).setConstant(200.0);

  jointDgain.setZero();
  jointDgain.tail(12).setConstant(10.0);

  jointVelocityTarget.setZero();

  jointConfig << 0, 0, 0.54, 1, 0, 0, 0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8,
      0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
  jointVel << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  auto anymal = world.addArticulatedSystem(exampleRscPath(argv[0], "anymal_c/urdf/anymal_sensored.urdf"));
  anymal->setState(jointConfig, jointVel);
  anymal->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  anymal->setPdGains(jointPgain, jointDgain);
  anymal->setPdTarget(jointConfig, jointVelocityTarget);
  anymal->setGeneralizedForce(Eigen::VectorXd::Zero(anymal->getDOF()));
  anymal->setName("Anymal");

  auto front_depthSensor = anymal->getSensorSet("depth_camera_front_camera_parent")->getSensor<raisim::DepthCamera>("depth");
  front_depthSensor->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);

  auto front_rgbCamera = anymal->getSensorSet("depth_camera_front_camera_parent")->getSensor<raisim::RGBCamera>("color");
  front_rgbCamera->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);

  auto rear_depthSensor = anymal->getSensorSet("depth_camera_rear_camera_parent")->getSensor<raisim::DepthCamera>("depth");
  rear_depthSensor->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);

  auto rear_rgbCamera = anymal->getSensorSet("depth_camera_rear_camera_parent")->getSensor<raisim::RGBCamera>("color");
  rear_rgbCamera->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);

  server.setupSocket();
  raisim_examples::warnIfNoClientConnected(server);
  while (!server.isConnected()) {
    server.acceptConnection(100000);
  }

  for (int k = 0; k < loopN; k++) {
    server.applyInteractionForce();
    world.integrate();
    if (server.waitForMessageFromClient(1)) {
      if (!server.processRequests()) {
        server.acceptConnection(100000);
      }
    }
  }

  server.closeConnection();
  return 0;
}
