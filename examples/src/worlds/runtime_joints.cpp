#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"

#include <cmath>
#include <iostream>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Joint = raisim::RuntimeJoint;

void fixture(raisim::World& world, double x, double y, double z,
             double sx, double sy, double sz) {
  auto* box = world.addBox(sx, sy, sz, 1., "default", 0, 0);
  box->setBodyType(raisim::BodyType::STATIC);
  box->setPosition(x, y, z);
  box->setAppearance("0.28, 0.34, 0.42, 1.0");
}

template<class Update>
void simulate(raisim::World& world, bool visualize,
              std::initializer_list<raisim::Object*> endpoints, Update update) {
  std::vector<raisim::World::ObjectId> endpointIds;
  for (auto* endpoint : endpoints) endpointIds.push_back(endpoint->getId());
  bool finished = false;
  std::unique_ptr<raisim::RaisimServer> server;
  if (visualize) {
    server = std::make_unique<raisim::RaisimServer>(&world);
    server->launchServer();
    std::cout << "Connect rayrai_tcp_viewer to port " << server->getPort()
              << ". The four-second scene repeats.\n" << std::flush;
  }
  int previousStep = -1;
  const auto beforeStep = [&] {
    for (auto id : endpointIds) {
      if (world.isObjectAlive(id)) continue;
      finished = true;
      std::cout << "A scripted endpoint was removed; the example is stopping.\n";
      return;
    }
    const int step = static_cast<int>(std::lround(world.getWorldTime() / world.getTimeStep()));
    if (step == previousStep) return;
    previousStep = step;
    update(step % 4000);
  };
  try {
    while (!finished && (server ? !server->isTerminateRequested() :
                        world.getWorldTime() < 4. - 0.5 * world.getTimeStep())) {
      if (server) {
        RS_TIMED_LOOP(1000)
        server->integrateWorldThreadSafe(beforeStep);
      } else {
        beforeStep();
        world.integrate();
      }
    }
  } catch (...) {
    if (server) server->killServer();
    throw;
  }
  if (server) server->killServer();
}

void toolDemo(raisim::World& world, bool visualize) {
  fixture(world, 0., 0.15, 1.42, 1.8, 0.12, 0.12);
  for (double x : {-0.85, 0.85}) fixture(world, x, 0.15, 0.7, 0.09, 0.12, 1.4);
  auto* wrist = world.addBox(0.28, 0.28, 0.18, 1.);
  wrist->setName("kinematic_wrist");
  wrist->setBodyType(raisim::BodyType::KINEMATIC);
  wrist->setAppearance("0.12, 0.45, 0.88, 1.0");
  auto* tool = world.addBox(0.16, 0.16, 0.4, 1.);
  tool->setName("detachable_tool");
  tool->setAppearance("1.0, 0.45, 0.08, 1.0");
  Joint* mount = nullptr;
  simulate(world, visualize, {wrist, tool}, [&](int step) {
    if (step == 0) {
      wrist->setPosition(-0.55, 0., 1.25);
      wrist->setOrientation(1., 0., 0., 0.);
      wrist->setAngularVelocity({0., 0., 0.});
      tool->setPosition(-0.55, 0., 0.96);
      tool->setOrientation(1., 0., 0., 0.);
      tool->setLinearVelocity({0., 0., 0.});
      tool->setAngularVelocity({0., 0., 0.});
      mount = world.addRuntimeJoint("tool_mount", Joint::Type::Fixed,
          {wrist, 0, {0., 0., -0.09}}, {tool, 0, {0., 0., 0.2}});
    }
    wrist->setLinearVelocity({step < 2000 ? 0.35 : -0.35, 0., 0.});
    if (step == 2000) {
      const double error = (wrist->getPosition() - tool->getPosition() -
                            Eigen::Vector3d(0., 0., 0.29)).norm();
      if (!visualize && (error > 1e-6 || !mount->getForce().e().allFinite() ||
                         std::abs(mount->getForce()[2] + 9.81) > 1e-6))
        throw std::runtime_error("the tool mount did not retain its pose and supported load");
      std::cout << "Support force on wrist: " << mount->getForce().e().transpose() << " N\n";
      world.removeRuntimeJoint(mount);
      mount = nullptr;
    }
  });
  if (!visualize) {
    std::cout << "Released tool position: " << tool->getPosition().transpose() << '\n';
    if (!tool->getPosition().allFinite() || tool->getPosition()[2] > 0.21 ||
        !world.getRuntimeJoints().empty())
      throw std::runtime_error("the released tool did not land on the ground");
  }
}

raisim::ArticulatedSystem* carriage(raisim::World& world, double y, const raisim::Vec<4>& color) {
  raisim::Body base(1., raisim::Mat<3, 3>::getIdentity(), {0., 0., 0.});
  raisim::Joint fixed;
  fixed.type = raisim::Joint::FIXED;
  fixed.name = "mount";
  fixed.jointPosition(raisim::Vec<3>{-0.8, y, 0.22});
  raisim::Child root(base, fixed, "base");
  raisim::Body load(1., raisim::Mat<3, 3>::getIdentity(), {0., 0., 0.});
  const auto addVisual = [&](const raisim::Vec<4>& size, const raisim::Vec<3>& position) {
    load.addVisualObject(raisim::Shape::Type::Box, size, position,
        raisim::Mat<3, 3>::getIdentity(), color, {1., 1., 1.}, "", "carriage", "");
  };
  addVisual({0.34, 0.22, 0.18, 0.}, {0., 0., 0.});
  addVisual({0.16, 0.30, 0.08, 0.}, {0., y < 0. ? 0.15 : -0.15, 0.15});
  raisim::Joint slide({1., 0., 0.}, {0., 0., 0.}, raisim::Mat<3, 3>::getIdentity(),
                      {-10., 10.}, raisim::Joint::PRISMATIC, "slide");
  root.addChild(raisim::Child(load, slide, "carriage"));
  auto* robot = world.addArticulatedSystem(root, "", 0, 0);
  robot->setName(y < 0. ? "driven_robot" : "passive_robot");
  return robot;
}

void dockingDemo(raisim::World& world, bool visualize) {
  for (double y : {-0.3, 0.3}) {
    fixture(world, 0.2, y, 0.08, 2.7, 0.25, 0.08);
    for (double x : {-1.1, 1.5}) fixture(world, x, y, 0.14, 0.08, 0.30, 0.20);
  }
  auto* first = carriage(world, -0.3, {1., 0.45, 0.08, 1.});
  auto* second = carriage(world, 0.3, {0.12, 0.45, 0.88, 1.});
  Joint* dock = nullptr;
  simulate(world, visualize, {first, second}, [&](int step) {
    if (step == 0) {
      for (auto* robot : {first, second}) {
        robot->setGeneralizedCoordinate(Eigen::VectorXd::Zero(1));
        robot->setGeneralizedVelocity(Eigen::VectorXd::Zero(1));
        robot->setGeneralizedForce(Eigen::VectorXd::Zero(1));
      }
      dock = world.addRuntimeJoint("dock", Joint::Type::Fixed,
          {first, first->getBodyIdx("carriage"), {0., 0.3, 0.15}},
          {second, second->getBodyIdx("carriage"), {0., -0.3, 0.15}});
    }
    first->setGeneralizedForce(Eigen::VectorXd::Constant(1, 0.35));
    if (step == 2000) {
      const double velocityError = std::abs(first->getGeneralizedVelocity()[0] -
                                            second->getGeneralizedVelocity()[0]);
      if (!visualize && (velocityError > 1e-10 || !dock->getForce().e().allFinite() ||
                         std::abs(dock->getForce()[0] + 0.175) > 1e-8))
        throw std::runtime_error("the robot dock did not transfer the applied load");
      std::cout << "Dock force on driven robot: " << dock->getForce().e().transpose() << " N\n";
      world.removeRuntimeJoint(dock);
      dock = nullptr;
    }
  });
  if (!visualize) {
    const double difference = first->getGeneralizedVelocity()[0] - second->getGeneralizedVelocity()[0];
    std::cout << "Released robot velocity difference: " << difference << " m/s\n";
    if (!std::isfinite(difference) || difference < 0.69 || !world.getRuntimeJoints().empty())
      throw std::runtime_error("the released robots did not move independently");
  }
}
}

int main(int argc, char** argv) {
  bool visualize = true;
  std::string scene = "tool", key;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--visualize") visualize = true;
    else if (argument == "--headless") visualize = false;
    else if ((argument == "--scene" || argument == "--activation-key") && i + 1 < argc) {
      (argument == "--scene" ? scene : key) = argv[++i];
    } else if (!argument.empty() && argument[0] != '-') key = argument;
    else throw std::invalid_argument("usage: runtime_joints [--headless|--visualize] [--scene tool|docking] [--activation-key FILE]");
  }
  if (scene != "tool" && scene != "docking") throw std::invalid_argument("scene must be tool or docking");
  if (!key.empty()) raisim::World::setActivationKey(key);
  raisim::World world;
  world.setTimeStep(0.001);
  world.setSleepingEnabled(false);
  world.addGround();
  if (scene == "tool") toolDemo(world, visualize);
  else dockingDemo(world, visualize);
}
