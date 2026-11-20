// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include <array>
#include <string>

#include "Eigen/Geometry"
#include "example_resources.hpp"
#include "raisim/RaisimServer.hpp"
#include "rayrai_tcp_viewer_hint.hpp"
#include "rocket_hop.hpp"

namespace {

void placeBehind(raisim::Visuals* visual, const Eigen::Vector3d& origin, const Eigen::Vector3d& axis, double offset, double radius,
                 double length) {
  const Eigen::Quaterniond facing = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), axis);
  visual->setPosition(origin - (offset + 0.5 * length) * axis);
  visual->setOrientation(facing.w(), facing.x(), facing.y(), facing.z());
  visual->setCylinderSize(radius, length);
}

}  // namespace

int main(int argc, char* argv[]) {
  namespace hop = raisim_examples::rocket_hop;
  raisim::World::setActivationKey(exampleRscPath(argv[0], "activation.raisim"));

  hop::Scene scene(exampleRscPath(argv[0], "rocket/rocket.urdf"));
  auto* rocket = scene.rocket();

  raisim::RaisimServer server(&scene.world());
  auto* flame = server.addVisualCylinder("flame", 0.17, 1.0, 1.0, 0.45, 0.1, 0.55);
  auto* core = server.addVisualCylinder("flame_core", 0.09, 1.0, 1.0, 0.9, 0.6, 0.9);
  std::array<raisim::Visuals*, 4> puffs{};
  for (std::size_t i = 0; i < puffs.size(); ++i)
    puffs[i] = server.addVisualCylinder("rcs_puff_" + std::to_string(i), 0.03, 0.2, 0.95, 0.95, 1.0, 0.7);
  for (const auto& pad : {hop::kPadA, hop::kPadB}) {
    auto* mark = server.addVisualCylinder("pad_mark_" + std::to_string(int(pad[0])), 1.4, 0.004, 0.95, 0.75, 0.1, 1.0);
    mark->setPosition(pad[0], pad[1], hop::kPadTop + 0.002);
  }
  server.launchServer();
  server.focusOn(rocket);
  raisim_examples::warnIfNoClientConnected(server);

  for (int i = 0; i < 10000000; ++i) {
    RS_TIMED_LOOP(int(hop::kTimeStep * 1e6))
    server.lockVisualizationServerMutex();
    scene.step();

    raisim::Vec<3> position, axis;
    rocket->getThrusterPose(scene.engine(), position, axis);
    const double throttle = scene.thrust() / hop::kMaxThrust;
    const double flameLength = throttle > 0.0 ? 0.8 + 3.0 * throttle : 0.0;
    placeBehind(flame, position.e(), axis.e(), hop::kNozzleLength, throttle > 0.0 ? 0.17 : 0.0, flameLength);
    placeBehind(core, position.e(), axis.e(), hop::kNozzleLength, throttle > 0.0 ? 0.09 : 0.0, 0.55 * flameLength);
    for (std::size_t k = 0; k < puffs.size(); ++k) {
      const double thrust = rocket->getThrust(scene.rollThrusters()[k]);
      rocket->getThrusterPose(scene.rollThrusters()[k], position, axis);
      placeBehind(puffs[k], position.e(), axis.e(), 0.04, thrust > 0.0 ? 0.03 : 0.0, 0.1 + 0.006 * thrust);
    }
    server.unlockVisualizationServerMutex();
  }

  server.killServer();
  return 0;
}
