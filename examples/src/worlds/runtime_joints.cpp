#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"

#include <cmath>
#include <iostream>
#include <initializer_list>
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

template<class Scene>
void simulate(raisim::World& world, Scene& scene) {
  std::vector<raisim::World::ObjectId> endpointIds;
  for (auto* endpoint : scene.endpoints()) endpointIds.push_back(endpoint->getId());
  bool finished = false;
  raisim::RaisimServer server(&world);
  server.launchServer();
  std::cout << "Connect rayrai_tcp_viewer to port " << server.getPort()
            << ". The four-second scene repeats.\n" << std::flush;
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
    scene.update(world, step % 4000);
  };
  try {
    while (!finished && !server.isTerminateRequested()) {
      RS_TIMED_LOOP(1000)
      server.integrateWorldThreadSafe(beforeStep);
    }
  } catch (...) {
    server.killServer();
    throw;
  }
  server.killServer();
}

/// A kinematic wrist carries a tool on a fixed runtime joint and releases it after two seconds.
struct ToolScene {
  raisim::Box* wrist = nullptr;
  raisim::Box* tool = nullptr;
  Joint* mount = nullptr;

  explicit ToolScene(raisim::World& world) {
    fixture(world, 0., 0.15, 1.42, 1.8, 0.12, 0.12);
    for (double x : {-0.85, 0.85}) fixture(world, x, 0.15, 0.7, 0.09, 0.12, 1.4);
    wrist = world.addBox(0.28, 0.28, 0.18, 1.);
    wrist->setName("kinematic_wrist");
    wrist->setBodyType(raisim::BodyType::KINEMATIC);
    wrist->setAppearance("0.12, 0.45, 0.88, 1.0");
    tool = world.addBox(0.16, 0.16, 0.4, 1.);
    tool->setName("detachable_tool");
    tool->setAppearance("1.0, 0.45, 0.08, 1.0");
  }

  std::vector<raisim::Object*> endpoints() const { return {wrist, tool}; }

  void update(raisim::World& world, int step) {
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
      std::cout << "Support force on wrist: " << mount->getForce().e().transpose() << " N\n";
      world.removeRuntimeJoint(mount);
      mount = nullptr;
    }
  }
};

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

/// Two independent robots dock through a fixed runtime joint, move together, and separate after two seconds.
struct DockingScene {
  raisim::ArticulatedSystem* first = nullptr;
  raisim::ArticulatedSystem* second = nullptr;
  Joint* dock = nullptr;

  explicit DockingScene(raisim::World& world) {
    for (double y : {-0.3, 0.3}) {
      fixture(world, 0.2, y, 0.08, 2.7, 0.25, 0.08);
      for (double x : {-1.1, 1.5}) fixture(world, x, y, 0.14, 0.08, 0.30, 0.20);
    }
    first = carriage(world, -0.3, {1., 0.45, 0.08, 1.});
    second = carriage(world, 0.3, {0.12, 0.45, 0.88, 1.});
  }

  std::vector<raisim::Object*> endpoints() const { return {first, second}; }

  void update(raisim::World& world, int step) {
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
      std::cout << "Dock force on driven robot: " << dock->getForce().e().transpose() << " N\n";
      world.removeRuntimeJoint(dock);
      dock = nullptr;
    }
  }
};
}

int main(int argc, char** argv) {
  std::string scene = "tool", key;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if ((argument == "--scene" || argument == "--activation-key") && i + 1 < argc) {
      (argument == "--scene" ? scene : key) = argv[++i];
    } else if (!argument.empty() && argument[0] != '-') key = argument;
    else throw std::invalid_argument("usage: runtime_joints [--scene tool|docking] [--activation-key FILE]");
  }
  if (scene != "tool" && scene != "docking") throw std::invalid_argument("scene must be tool or docking");
  if (!key.empty()) raisim::World::setActivationKey(key);
  raisim::World world;
  world.setTimeStep(0.001);
  world.setSleepingEnabled(false);
  world.addGround();
  if (scene == "tool") {
    ToolScene tool(world);
    simulate(world, tool);
  } else {
    DockingScene docking(world);
    simulate(world, docking);
  }
  return 0;
}
