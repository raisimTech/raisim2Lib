// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include <array>
#include <string>

#include "Eigen/Geometry"
#include "example_resources.hpp"
#include "magnetic_wall_climber.hpp"
#include "raisim/RaisimServer.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

int main(int argc, char* argv[]) {
  namespace climber = raisim_examples::wall_climber;
  raisim::World::setActivationKey(exampleRscPath(argv[0], "activation.raisim"));

  climber::Scene scene(exampleRscPath(argv[0], "go1/go1.urdf"));
  auto* robot = scene.robot();

  raisim::RaisimServer server(&scene.world());
  for (int row = 1; row < int(climber::kWallHeight); ++row)
    server.addVisualBox("seam_row_" + std::to_string(row), 0.002, climber::kWallWidth, 0.012, 0.25, 0.27, 0.3, 1.0)
        ->setPosition(0.0, 0.0, row);
  for (int column = 1; column < int(climber::kWallWidth); ++column)
    server.addVisualBox("seam_column_" + std::to_string(column), 0.002, 0.012, climber::kWallHeight, 0.25, 0.27, 0.3, 1.0)
        ->setPosition(0.0, column - 0.5 * climber::kWallWidth, 0.5 * climber::kWallHeight);
  std::array<raisim::Visuals*, 4> magnets{};
  for (std::size_t leg = 0; leg < magnets.size(); ++leg)
    magnets[leg] = server.addVisualCylinder(std::string(climber::kLegNames[leg]) + "_magnet", 0.03, 0.012);
  server.launchServer();
  // The viewer's follow camera would look from behind the wall; look at the sunlit face instead.
  server.setCameraPositionAndLookAt({-4.4, 2.6, 2.6}, {0.0, 0.0, 1.9});
  raisim_examples::warnIfNoClientConnected(server);

  for (int i = 0; i < 10000000; ++i) {
    RS_TIMED_LOOP(int(climber::kTimeStep * 1e6))
    server.lockVisualizationServerMutex();
    scene.step();
    for (std::size_t leg = 0; leg < magnets.size(); ++leg) {
      raisim::Vec<3> pad, axis;
      robot->getGripperPose(scene.magnet(leg), pad, axis);
      const Eigen::Quaterniond facing = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), axis.e());
      magnets[leg]->setPosition(pad.e() - 0.006 * axis.e());
      magnets[leg]->setOrientation(facing.w(), facing.x(), facing.y(), facing.z());
      if (scene.magnetHolds(leg))
        magnets[leg]->setColor(0.9, 0.1, 0.1, 1.0);
      else if (scene.magnetOn(leg))
        magnets[leg]->setColor(1.0, 0.6, 0.1, 1.0);
      else
        magnets[leg]->setColor(0.35, 0.35, 0.38, 1.0);
    }
    server.unlockVisualizationServerMutex();
  }

  server.killServer();
  return 0;
}
