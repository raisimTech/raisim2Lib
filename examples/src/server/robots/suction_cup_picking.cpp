// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "Eigen/Geometry"
#include "example_resources.hpp"
#include "raisim/RaisimServer.hpp"
#include "rayrai_tcp_viewer_hint.hpp"
#include "suction_picking.hpp"

int main(int argc, char* argv[]) {
  namespace picking = raisim_examples::suction_picking;
  raisim::World::setActivationKey(exampleRscPath(argv[0], "activation.raisim"));

  picking::Scene scene(exampleRscPath(argv[0], "kinova_suction/kinova_suction.urdf"));
  auto* robot = scene.robot();

  raisim::RaisimServer server(&scene.world());
  auto* ring = server.addVisualCylinder("vacuum_ring", 0.027, 0.006, 0.2, 0.85, 0.3, 1.0);
  server.launchServer();
  server.focusOn(robot);
  raisim_examples::warnIfNoClientConnected(server);

  for (int i = 0; i < 10000000; ++i) {
    RS_TIMED_LOOP(int(picking::kTimeStep * 1e6))
    server.lockVisualizationServerMutex();
    scene.step();
    raisim::Vec<3> pad, axis;
    robot->getGripperPose(scene.cup(), pad, axis);
    const Eigen::Quaterniond facing = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), axis.e());
    ring->setPosition(pad.e() - 0.012 * axis.e());
    ring->setOrientation(facing.w(), facing.x(), facing.y(), facing.z());
    ring->setCylinderSize(scene.suctionOn() ? 0.027 : 0.0, 0.006);
    if (scene.holding())
      ring->setColor(0.2, 0.85, 0.3, 1.0);
    else
      ring->setColor(1.0, 0.6, 0.1, 1.0);
    server.unlockVisualizationServerMutex();
  }

  server.killServer();
  return 0;
}
