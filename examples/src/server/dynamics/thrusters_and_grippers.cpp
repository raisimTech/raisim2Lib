// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include <algorithm>
#include <cmath>
#include <string>

#include "raisim/RaisimServer.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

namespace {

constexpr double kMass = 1.5, kArm = 0.2, kDrag = 0.016;

// A quadrotor in the plus layout: the front and back propellers spin one way, left and right the other.
std::string quadrotor() {
  std::string urdf =
      "<robot name=\"quadrotor\"><link name=\"base\">"
      "<inertial><mass value=\"1.5\"/><inertia ixx=\"0.01\" iyy=\"0.01\" izz=\"0.02\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/></inertial>"
      "<visual><geometry><box size=\"0.44 0.04 0.02\"/></geometry></visual>"
      "<visual><geometry><box size=\"0.04 0.44 0.02\"/></geometry></visual>"
      "<collision><geometry><box size=\"0.44 0.44 0.04\"/></geometry></collision></link>";
  const char* names[4] = {"front", "back", "left", "right"};
  const double x[4] = {kArm, -kArm, 0., 0.}, y[4] = {0., 0., kArm, -kArm};
  for (int i = 0; i < 4; ++i) {
    const std::string at = std::to_string(x[i]) + " " + std::to_string(y[i]) + " 0.02";
    urdf += std::string("<thruster name=\"") + names[i] + "\" link=\"base\" torque_coefficient=\"" +
            std::to_string(i < 2 ? kDrag : -kDrag) + "\"><origin xyz=\"" + at +
            "\"/><limit lower=\"0\" upper=\"15\"/></thruster>";
  }
  return urdf + "</robot>";
}

// A tool on a vertical slide, with a suction cup or a magnet under it.
std::string lifter(const std::string& type) {
  return "<robot name=\"lifter\"><link name=\"world\"/>"
         "<joint name=\"mount\" type=\"fixed\"><parent link=\"world\"/><child link=\"base\"/></joint>"
         "<link name=\"base\"><inertial><mass value=\"1\"/><inertia ixx=\"0.01\" iyy=\"0.01\" izz=\"0.01\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/></inertial></link>"
         "<joint name=\"slide\" type=\"prismatic\"><parent link=\"base\"/><child link=\"tool\"/><axis xyz=\"0 0 1\"/>"
         "<limit lower=\"-1\" upper=\"2\" effort=\"1000\" velocity=\"10\"/></joint>"
         "<link name=\"tool\"><inertial><mass value=\"1\"/><inertia ixx=\"0.01\" iyy=\"0.01\" izz=\"0.01\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/></inertial>"
         "<visual><geometry><cylinder radius=\"0.04\" length=\"0.02\"/></geometry></visual>"
         "<visual><origin xyz=\"0 0 0.3\"/><geometry><cylinder radius=\"0.01\" length=\"0.6\"/></geometry></visual>"
         "<collision><geometry><cylinder radius=\"0.04\" length=\"0.02\"/></geometry></collision></link>"
         "<gripper name=\"pad\" link=\"tool\" type=\"" + type + "\" max_force=\"40\" range=\"0.01\">"
         "<origin xyz=\"0 0 -0.01\"/><axis xyz=\"0 0 -1\"/></gripper></robot>";
}

// Thrusts that give the total thrust and the body torques (roll, pitch, yaw) of the plus layout.
Eigen::VectorXd mix(double thrust, const Eigen::Vector3d& torque) {
  Eigen::VectorXd thrusts(4);
  thrusts << thrust / 4. - torque.y() / (2. * kArm) + torque.z() / (4. * kDrag),
      thrust / 4. + torque.y() / (2. * kArm) + torque.z() / (4. * kDrag),
      thrust / 4. + torque.x() / (2. * kArm) - torque.z() / (4. * kDrag),
      thrust / 4. - torque.x() / (2. * kArm) - torque.z() / (4. * kDrag);
  return thrusts;
}

}  // namespace

int main() {
  raisim::World world;
  world.setTimeStep(0.001);
  world.addGround();
  world.setMagneticMaterial("steel");

  auto* quad = world.addArticulatedSystem(quadrotor());
  quad->setName("quadrotor");
  Eigen::VectorXd gc(7);
  gc << 0., 0., 0.2, 1., 0., 0., 0.;
  quad->setGeneralizedCoordinate(gc);

  auto* cupLifter = world.addArticulatedSystem(lifter("suction"));
  cupLifter->setName("suction lifter");
  cupLifter->setBasePos(raisim::Vec<3>{1., -0.5, 0.});
  auto* box = world.addBox(0.12, 0.12, 0.12, 1.);
  box->setPosition(1., -0.5, 0.06);
  box->setAppearance("0.9, 0.5, 0.2, 1.0");

  auto* magnetLifter = world.addArticulatedSystem(lifter("magnetic"));
  magnetLifter->setName("magnet lifter");
  magnetLifter->setBasePos(raisim::Vec<3>{1., 0.5, 0.});
  auto* steel = world.addBox(0.2, 0.06, 0.04, 2., "steel");
  steel->setPosition(1., 0.5, 0.02);
  steel->setAppearance("0.6, 0.6, 0.65, 1.0");

  for (auto* robot : {cupLifter, magnetLifter})
    robot->setPdGains(Eigen::VectorXd::Constant(1, 20000.), Eigen::VectorXd::Constant(1, 400.));
  cupLifter->setGeneralizedCoordinate({0.13});
  magnetLifter->setGeneralizedCoordinate({0.052});

  raisim::RaisimServer server(&world);
  server.launchServer();
  server.focusOn(quad);
  raisim_examples::warnIfNoClientConnected(server);

  for (int i = 0; i < 60000; i++) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    const double t = world.getWorldTime();

    // the quadrotor climbs to 1.5 m, bobs up and down and turns slowly
    const Eigen::Matrix3d rotation = quad->getBaseOrientation().e();
    const auto& u = quad->getGeneralizedVelocity().e();
    const Eigen::Vector3d omega = rotation.transpose() * u.segment<3>(3);
    const double height = 1.5 + 0.3 * std::sin(0.5 * t), yaw = 0.8 * std::sin(0.3 * t);
    const double yawError = std::remainder(yaw - std::atan2(rotation(1, 0), rotation(0, 0)), 2. * M_PI);
    const double vertical = kMass * (9.81 + 8. * (height - quad->getGeneralizedCoordinate()[2]) - 5. * u[2]);
    const Eigen::Vector3d torque(-2. * rotation(2, 1) - 0.3 * omega.x(),
                                 2. * rotation(2, 0) - 0.3 * omega.y(),
                                 0.5 * yawError - 0.1 * omega.z());
    quad->setThrusts(mix(vertical / rotation(2, 2), torque));

    // the lifters lower, grip, lift for two seconds and let go, every five seconds
    const double phase = std::fmod(t, 5.);
    const bool grip = phase > 0.5 && phase < 3.;
    const double rise = std::clamp(0.2 * (phase - 1.), 0., 0.3) - std::clamp(0.3 * (phase - 3.5), 0., 0.3);
    cupLifter->setGripperActive(0, grip);
    magnetLifter->setGripperActive(0, grip);
    cupLifter->setPdTarget(Eigen::VectorXd::Constant(1, 0.13 + rise), Eigen::VectorXd::Zero(1));
    magnetLifter->setPdTarget(Eigen::VectorXd::Constant(1, 0.052 + rise), Eigen::VectorXd::Zero(1));
    server.integrateWorldThreadSafe();
  }

  server.killServer();
  return 0;
}
