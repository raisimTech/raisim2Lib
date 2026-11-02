// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "example_resources.hpp"
#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main(int argc, char* argv[]) {
  raisim::World world;
  world.setTimeStep(0.001);

  /// create raisim environment
  world.addGround();
  auto box1 = world.addBox(1, 3, 0.3, 100);
  box1->setPosition(2, 0, 0.15);
  auto box2 = world.addBox(1, 3, 0.2, 100);
  box2->setPosition(4, 0, 0.1);
  auto box3 = world.addBox(1, 3, 0.3, 100);
  box3->setPosition(6, 0, 0.15);
  auto box4 = world.addBox(1, 3, 0.2, 100);
  box4->setPosition(8, 0, 0.1);
  auto box5 = world.addBox(1, 3, 0.3, 100);
  box5->setPosition(10, 0, 0.15);

  /// customize robot in code: each @@name in the URDF template is replaced by its value
  std::vector<raisim::World::ParameterContainer> params;
  params.push_back({"rho", "500"});
  params.push_back({"base_length", "1.0"});
  params.push_back({"base_width", "0.15"});
  params.push_back({"wheel_y_offset", "0.14"});
  params.push_back({"track_y_offset", "0.24"});
  params.push_back({"wheel_width", "0.08"});
  params.push_back({"wheel_radius", "0.17775"});
  params.push_back({"flipper_radius", "0.16"});
  params.push_back({"track_thickness", "0.015"});

  raisim::ArticulatedSystemOption options;
  options.convexifyCollisionMeshes = true;
  auto robot = world.addArticulatedSystem(exampleRscPath(argv[0], "templatedTrackedRobot/trackedTemplate.urdf"), params, "", {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);
  robot->setName("tracked");
  robot->setBasePos({0, 0, 0.25});

  /// the belts move with the sprockets (see the <track> elements of the URDF): drive the sprockets
  /// at 5 rad/s and hold the flippers raised at 0.6 rad
  Eigen::VectorXd pGain = Eigen::VectorXd::Zero(robot->getDOF()), dGain = pGain, dTarget = pGain;
  Eigen::VectorXd pTarget = Eigen::VectorXd::Zero(robot->getGeneralizedCoordinateDim());
  for (const char* sprocket : {"left_sprocket", "right_sprocket"}) {
    const size_t gv = robot->getGeneralizedVelocityIndex(sprocket);
    dGain[gv] = 50.0;
    dTarget[gv] = 5.0;
  }
  for (const char* flipper : {"left_flipper_joint", "right_flipper_joint"}) {
    pGain[robot->getGeneralizedVelocityIndex(flipper)] = 250.0;
    dGain[robot->getGeneralizedVelocityIndex(flipper)] = 50.0;
    pTarget[robot->getJoint(flipper).getIdxInGeneralizedCoordinate()] = -0.6;
  }
  robot->setPdGains(pGain, dGain);
  robot->setPdTarget(pTarget, dTarget);

  /// launch raisim server
  raisim::RaisimServer server(&world);
  server.launchServer();
  raisim_examples::warnIfNoClientConnected(server);
  server.focusOn(robot);

  for (int i = 0; i < 200000; i++) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    server.integrateWorldThreadSafe();
  }

  server.killServer();
  return 0;
}
