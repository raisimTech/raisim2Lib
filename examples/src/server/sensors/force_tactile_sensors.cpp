// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "example_resources.hpp"
#include "raisim/RaisimServer.hpp"
#include "raisim/sensors/ForceSensor.hpp"
#include "raisim/sensors/TactileSensor.hpp"

int main(int argc, char** argv) {
  try {
    int steps = 5000;
    bool serve = false;
    std::string key = exampleRscPath(argv[0], "activation.raisim");
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--server") serve = true;
      else if (argument.rfind("--steps=", 0) == 0) steps = std::stoi(argument.substr(8));
      else if (argument.rfind("--activation-key=", 0) == 0) key = argument.substr(17);
      else throw std::invalid_argument("unknown argument: " + argument);
    }
    if (steps <= 0) throw std::invalid_argument("steps must be positive");
    raisim::World::setActivationKey(key);
    raisim::World world;
    world.setTimeStep(0.001);
    world.setDefaultMaterial(0.2, 0., 0.);
    auto* robot = world.addArticulatedSystem(std::string(R"xml(
      <robot name="instrumented_pad"><link name="world"/>
        <link name="pad"><inertial><mass value="1"/>
          <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
          <collision><geometry><box size="0.8 0.8 0.2"/></geometry></collision>
          <visual><geometry><box size="0.8 0.8 0.2"/></geometry></visual>
          <sensor name="mount_load" type="force" update_rate="500"/>
          <sensor name="surface" type="tactile" update_rate="500" width="0.8" height="0.8"
                  depth="0.01" rows="8" columns="8">
            <origin xyz="0 0 0.1" rpy="0 -1.5707963267948966 0"/>
          </sensor>
        </link>
        <joint name="load_cell" type="fixed"><parent link="world"/><child link="pad"/></joint>
      </robot>)xml"));
    robot->setName("instrumented_pad");
    auto* ball = world.addSphere(0.08, 2.);
    ball->setPosition(0.1, 0.1, 0.18);
    ball->setName("load");
    auto* force = robot->getSensorSet("pad")->getSensor<raisim::ForceSensor>("mount_load");
    auto* tactile = robot->getSensorSet("pad")->getSensor<raisim::TactileSensor>("surface");
    std::unique_ptr<raisim::RaisimServer> server;
    if (serve) {
      server = std::make_unique<raisim::RaisimServer>(&world);
      server->launchServer();
      server->focusOn(robot);
    }
    std::cout << "time,joint_force_z,pad_normal_force,center_y,center_z,slip_speed\n";
    std::cout << std::setprecision(10);
    for (int step = 0; step < steps; ++step) {
      if (serve) { RS_TIMED_LOOP(1000) }
      if (world.getWorldTime() > 1.) {
        ball->setExternalForce(0, {0., 7. * std::sin(3. * world.getWorldTime()), 0.});
        if (std::abs(ball->getPosition()[1]) > 0.28) {
          ball->setPosition(0.1, 0., 0.18);
          ball->setVelocity(0., 0., 0., 0., 0., 0.);
        }
      }
      if (server) server->integrateWorldThreadSafe();
      else world.integrate();
      if (step % 20 == 0) {
        double slip = 0.;
        for (const auto& velocity : tactile->getTaxelSlipVelocities()) slip = std::max(slip, velocity.norm());
        std::cout << world.getWorldTime() << ',' << force->getForce()[2] << ',' << tactile->getNormalForce()
            << ',' << tactile->getCenterOfPressure()[0] << ',' << tactile->getCenterOfPressure()[1]
            << ',' << slip << '\n';
      }
    }
    if (server) server->killServer();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
