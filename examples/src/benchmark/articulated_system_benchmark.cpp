// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "benchmarkCommon.hpp"

#include <Eigen/Dense>
#include <raisim/World.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace {

raisim::Path findRscDir(const raisim::Path& binaryDir) {
  // CMake copies `rsc/` next to the examples binaries on Windows.
  // On other platforms, output directory layouts can vary (depending on the parent build).
  const std::vector<std::string> suffixes = {
      "/rsc",
      "/../rsc",
      "/../../rsc",
      "/../../../rsc",
  };

  for (const auto& s : suffixes) {
    raisim::Path candidate(binaryDir.getString() + s);
    if (candidate.directoryExists()) return candidate;
  }

  std::cerr << "[articulated_system_benchmark] Could not find `rsc` directory. Tried:";
  for (const auto& s : suffixes) std::cerr << " " << (binaryDir.getString() + s);
  std::cerr << std::endl;
  return raisim::Path("");
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);
  const raisim::Path binaryDir = binaryPath.getDirectory();
  const raisim::Path rscDir = findRscDir(binaryDir);
  if (rscDir.getString().empty()) return 1;

  raisim::World::setActivationKey(rscDir.getString() + "/activation.raisim");

  int loopN = 1000000;
  std::chrono::steady_clock::time_point begin, end;

  raisim::World world;
  world.setSleepingEnabled(false);

  auto* checkerBoard = world.addGround();

  Eigen::VectorXd jointConfig(19), jointVelocityTarget(18);
  Eigen::VectorXd jointVel(18), jointPgain(18), jointDgain(18);

  jointPgain.setZero();
  jointPgain.tail(12).setConstant(200.0);

  jointDgain.setZero();
  jointDgain.tail(12).setConstant(10.0);

  jointVelocityTarget.setZero();

  jointConfig << 0, 0, 0.54, 1, 0, 0, 0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4,
      0.8, -0.03, -0.4, 0.8;
  jointVel.setZero();

  auto anymal = world.addArticulatedSystem(rscDir.getString() + "/anymal/urdf/anymal.urdf");
  anymal->setState(jointConfig, jointVel);
  anymal->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  anymal->setPdGains(jointPgain, jointDgain);
  anymal->setPdTarget(jointConfig, jointVelocityTarget);
  anymal->setGeneralizedForce(Eigen::VectorXd::Zero(anymal->getDOF()));

  begin = std::chrono::steady_clock::now();

  for (int i = 0; i < loopN; i++) {
    world.integrate();
  }

  end = std::chrono::steady_clock::now();

  raisim::print_timediff("ANYmal, 4 contacts", loopN, begin, end);

  world.removeObject(checkerBoard);
  begin = std::chrono::steady_clock::now();

  for (int i = 0; i < loopN; i++) {
    world.integrate();
  }

  end = std::chrono::steady_clock::now();

  raisim::print_timediff("ANYmal, 0 contact", loopN, begin, end);

  /////////////////////////// atlas ////////////////////////////////////
  raisim::World world2;
  world2.setTimeStep(0.001);

  loopN = 100000;

  /// create objects
  auto* ground2 = world2.addGround();
  auto atlas = world2.addArticulatedSystem(rscDir.getString() + "/atlas/robot.urdf");
  atlas->setGeneralizedCoordinate({0, 0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  atlas->setGeneralizedForce(Eigen::VectorXd::Zero(atlas->getDOF()));

  size_t contactSize = 0;
  begin = std::chrono::steady_clock::now();

  for (int i = 0; i < loopN; i++) {
    world2.integrate();
    contactSize += world2.getContactProblem()->size();
  }

  end = std::chrono::steady_clock::now();

  raisim::print_timediff("Atlas", loopN, begin, end);
  std::cout << "average number of contacts " << double(contactSize) / loopN << std::endl;

  world2.removeObject(ground2);
  begin = std::chrono::steady_clock::now();
  contactSize = 0;
  for (int i = 0; i < loopN; i++) {
    world2.integrate();
    contactSize += world2.getContactProblem()->size();
  }

  end = std::chrono::steady_clock::now();

  raisim::print_timediff("Atlas, no ground", loopN, begin, end);
  std::cout << "average number of contacts " << double(contactSize) / loopN << std::endl;

  //////////////////////////////////// chain 10 /////////////////////////////////////////////////////
  raisim::World world3;
  world3.setTimeStep(0.001);

  loopN = 100000;

  /// create objects
  world3.addArticulatedSystem(rscDir.getString() + "/chain/robot_springed_10.urdf");

  begin = std::chrono::steady_clock::now();

  for (int i = 0; i < loopN; i++) world3.integrate();

  end = std::chrono::steady_clock::now();

  raisim::print_timediff("Chain10 (30 dof)", loopN, begin, end);

  raisim::World world4;
  world4.setTimeStep(0.001);

  loopN = 100000;

  /// create objects
  world4.addArticulatedSystem(rscDir.getString() + "/chain/robot_springed_20.urdf");

  begin = std::chrono::steady_clock::now();

  for (int i = 0; i < loopN; i++) world4.integrate();

  end = std::chrono::steady_clock::now();

  raisim::print_timediff("Chain20 (60 dof)", loopN, begin, end);

  return 0;
}
