// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kCycle = 6000;

raisim::Joint makeJoint(raisim::Joint::Type type, const std::string& name, const Eigen::Vector3d& position,
                        const Eigen::Vector3d& axis = Eigen::Vector3d::UnitX()) {
  raisim::Joint joint;
  joint.type = type;
  joint.name = name;
  joint.pos_P.e() = position;
  joint.axis.e() = axis;
  return joint;
}

Eigen::Vector3d pointInBody(raisim::ArticulatedSystem& system, size_t body, const Eigen::Vector3d& point) {
  raisim::Mat<3, 3> rotation;
  raisim::Vec<3> position;
  system.getBodyPose(body, rotation, position);
  return rotation.e().transpose() * (point - position.e());
}

void removeIfAlive(raisim::World& world, raisim::Object*& object) {
  if (object && world.isObjectAlive(object->getId())) world.removeObject(object);
  object = nullptr;
}

raisim::Box* addBox(raisim::World& world, const std::string& name, const Eigen::Vector3d& size, double mass,
                    const Eigen::Vector3d& position, const std::string& color) {
  auto* box = world.addBox(size.x(), size.y(), size.z(), mass);
  box->setName(name);
  box->setPosition(position);
  box->setAppearance(color);
  return box;
}

void addBoxGeometry(raisim::Body& body, const std::string& name, const raisim::Vec<4>& size,
                    const raisim::Vec<3>& offset, const raisim::Vec<4>& color, bool collide) {
  if (collide)
    body.addCollisionObject(raisim::Shape::Box, size, offset, raisim::Mat<3, 3>::getIdentity(), {1., 1., 1.}, name,
                            "default", "", "");
  body.addVisualObject(raisim::Shape::Box, size, offset, raisim::Mat<3, 3>::getIdentity(), color, {1., 1., 1.}, "",
                       name, "");
}

/// Station A: a static beam and four boxes are joined one by one into a chain that swings.
struct ChainStation {
  double x = -1.5;
  raisim::Object* chain = nullptr;

  void reset(raisim::World& world) {
    removeIfAlive(world, chain);
    auto* beam = addBox(world, "beam", {0.5, 0.12, 0.08}, 1., {x, 0., 1.7}, "0.28, 0.34, 0.42, 1.0");
    beam->setBodyType(raisim::BodyType::STATIC);
    raisim::Object* parent = beam;
    size_t parentBody = 0;
    Eigen::Vector3d hinge(x, 0., 1.66);
    for (int i = 0; i < 4; ++i) {
      const std::string name = "link" + std::to_string(i);
      auto* link = addBox(world, name, {0.06, 0.06, 0.25}, 0.5, hinge - Eigen::Vector3d(0., 0., 0.125),
                          i % 2 ? "1.0, 0.45, 0.08, 1.0" : "0.12, 0.45, 0.88, 1.0");
      link->setVelocity(Eigen::Vector3d(0.9 * (i + 1), 0., 0.), Eigen::Vector3d::Zero());
      Eigen::Vector3d position = hinge - Eigen::Vector3d(x, 0., 1.7);
      if (i > 0) position = pointInBody(*static_cast<raisim::ArticulatedSystem*>(parent), parentBody, hinge);
      auto* system = world.joinObjects(parent, parentBody, link, 0,
                                       makeJoint(raisim::Joint::REVOLUTE, "hinge" + std::to_string(i), position,
                                                 Eigen::Vector3d::UnitY()));
      system->setName("chain");
      parent = system;
      parentBody = system->getBodyIdx(name);
      hinge.z() -= 0.25;
    }
    chain = parent;
  }

  void check(raisim::World& world) const {
    if (!world.isObjectAlive(chain->getId())) throw std::runtime_error("the chain was removed");
    auto* system = static_cast<raisim::ArticulatedSystem*>(chain);
    const auto& gc = system->getGeneralizedCoordinate();
    if (system->getDOF() != 4 || !gc.e().allFinite() || gc.e().cwiseAbs().maxCoeff() < 0.1)
      throw std::runtime_error("the joined chain did not swing as a four-joint pendulum");
    std::cout << "Chain joint angles: " << gc.e().transpose() << " rad" << std::endl;
  }
};

/// Station B: a gantry lowers its hook onto a tool, joins it, carries it and releases it.
struct GantryStation {
  double x = 0.;
  raisim::Object* robot = nullptr;
  raisim::Object* tool = nullptr;
  Eigen::Vector3d carriedTool = Eigen::Vector3d::Zero();

  void reset(raisim::World& world) {
    removeIfAlive(world, robot);
    removeIfAlive(world, tool);
    raisim::Body frame(5., raisim::Mat<3, 3>::getIdentity(), {0., 0., 0.});
    frame.setInertia(raisim::Vec<3>{1., 1., 1.});
    const raisim::Vec<4> steel = {0.28, 0.34, 0.42, 1.};
    addBoxGeometry(frame, "post_left", {0.06, 0.06, 1.4, 0.}, {-0.7, 0.3, -0.7}, steel, false);
    addBoxGeometry(frame, "post_right", {0.06, 0.06, 1.4, 0.}, {0.7, 0.3, -0.7}, steel, false);
    addBoxGeometry(frame, "beam", {1.46, 0.06, 0.06, 0.}, {0., 0.3, 0.}, steel, false);
    raisim::Joint mount;
    mount.type = raisim::Joint::FIXED;
    mount.name = "gantry_mount";
    mount.pos_P = {x, -0.3, 1.4};
    raisim::Child root(frame, mount, "frame");

    raisim::Body carriage(1., raisim::Mat<3, 3>::getIdentity(), {0., 0., 0.});
    carriage.setInertia(raisim::Vec<3>{0.01, 0.01, 0.01});
    addBoxGeometry(carriage, "carriage", {0.16, 0.12, 0.1, 0.}, {0., 0.3, -0.08}, {0.12, 0.45, 0.88, 1.}, false);
    raisim::Child carriageNode(carriage, raisim::Joint({1., 0., 0.}, {0., 0., 0.}, raisim::Mat<3, 3>::getIdentity(),
                                                        {-0.6, 0.6}, raisim::Joint::PRISMATIC, "travel"), "carriage");
    raisim::Body hook(0.5, raisim::Mat<3, 3>::getIdentity(), {0., 0., 0.});
    hook.setInertia(raisim::Vec<3>{0.005, 0.005, 0.005});
    addBoxGeometry(hook, "rod", {0.03, 0.03, 0.6, 0.}, {0., 0.3, 0.2}, {0.6, 0.6, 0.6, 1.}, false);
    addBoxGeometry(hook, "pad", {0.12, 0.12, 0.04, 0.}, {0., 0.3, -0.12}, {0.12, 0.45, 0.88, 1.}, true);
    carriageNode.addChild(raisim::Child(hook, raisim::Joint({0., 0., 1.}, {0., 0., -0.4}, raisim::Mat<3, 3>::getIdentity(),
                                                            {-0.8, 0.2}, raisim::Joint::PRISMATIC, "lift"), "hook"));
    root.addChild(carriageNode);
    auto* gantry = world.addArticulatedSystem(root);
    gantry->setName("gantry");
    gantry->setPdGains(Eigen::Vector2d(800., 3000.), Eigen::Vector2d(80., 200.));
    gantry->setPdTarget(Eigen::Vector2d(-0.4, 0.), Eigen::Vector2d::Zero());
    gantry->setGeneralizedCoordinate(Eigen::Vector2d(-0.4, 0.));
    robot = gantry;
    tool = addBox(world, "tool", {0.12, 0.12, 0.16}, 0.4, {x - 0.4, 0., 0.08}, "1.0, 0.45, 0.08, 1.0");
  }

  void update(raisim::World& world, int step, bool checks) {
    auto* gantry = static_cast<raisim::ArticulatedSystem*>(robot);
    const double t = step * world.getTimeStep();
    const auto smooth = [](double s) { s = std::clamp(s, 0., 1.); return s * s * (3. - 2. * s); };
    double travel = -0.4, lift = 0.;
    const double grasp = -0.698;
    if (t < 1.) lift = grasp * smooth(t);
    else if (t < 1.8) lift = grasp * (1. - smooth(t - 1.));
    else lift = 0.;
    if (t >= 1.8) travel = -0.4 + 0.8 * smooth((t - 1.8) / 1.4);
    Eigen::VectorXd target(gantry->getGeneralizedCoordinateDim());
    target.setZero();
    target[gantry->getJoint("travel").getIdxInGeneralizedCoordinate()] = travel;
    target[gantry->getJoint("lift").getIdxInGeneralizedCoordinate()] = lift;
    gantry->setPdTarget(target, Eigen::VectorXd::Zero(gantry->getDOF()));

    if (step == 1000) {
      const size_t hook = gantry->getBodyIdx("hook");
      const Eigen::Vector3d grip = static_cast<raisim::SingleBodyObject*>(tool)->getPosition() +
          Eigen::Vector3d(0., 0., 0.08);
      robot = world.joinObjects(gantry, hook, tool, 0,
                                makeJoint(raisim::Joint::FIXED, "tool_mount", pointInBody(*gantry, hook, grip)));
      tool = nullptr;
    }
    if (step == 3599) {
      raisim::Vec<3> grip;
      static_cast<raisim::ArticulatedSystem*>(robot)->getFramePosition("tool_mount", grip);
      carriedTool = grip.e() - Eigen::Vector3d(0., 0., 0.08);
    }
    if (step == 3600) {
      auto parts = world.detachJoint(static_cast<raisim::ArticulatedSystem*>(robot), "tool_mount");
      robot = parts[0];
      tool = parts[1];
    }
  }

  void check(raisim::World& world) const {
    auto* released = static_cast<raisim::ArticulatedSystem*>(tool);
    const raisim::Vec<3> position = released->getBodyCOM_W()[0];
    std::cout << "Tool carried to " << carriedTool.transpose() << " and released; it rests at "
              << position.e().transpose() << std::endl;
    if (std::abs(carriedTool.x() - (x + 0.4)) > 0.02 || carriedTool.z() < 0.6 || std::abs(position[0] - (x + 0.4)) > 0.05 ||
        std::abs(position[2] - 0.08) > 0.01)
      throw std::runtime_error("the gantry did not carry and release the tool");
  }
};

/// Station C: two fixed-base carriages dock through a loop joint, move together, and undock.
struct DockingStation {
  double x = 1.5;
  raisim::Object* pair = nullptr;
  double dockedError = 0.;
  std::vector<std::string> loopJoints;

  raisim::ArticulatedSystem* carriage(raisim::World& world, const std::string& name, double y,
                                      const raisim::Vec<4>& color) {
    raisim::Body rail(1., raisim::Mat<3, 3>::getIdentity(), {0., 0., 0.});
    rail.setInertia(raisim::Vec<3>{0.1, 0.1, 0.1});
    addBoxGeometry(rail, "rail", {2.6, 0.08, 0.04, 0.}, {1.05, 0., -0.1}, {0.28, 0.34, 0.42, 1.}, false);
    raisim::Joint mount;
    mount.type = raisim::Joint::FIXED;
    mount.name = "rail_mount";
    mount.pos_P = {x - 0.4, y, 0.25};
    raisim::Child root(rail, mount, "rail");
    raisim::Body cart(1., raisim::Mat<3, 3>::getIdentity(), {0., 0., 0.});
    cart.setInertia(raisim::Vec<3>{0.01, 0.01, 0.01});
    addBoxGeometry(cart, "cart", {0.3, 0.2, 0.14, 0.}, {0., 0., 0.}, color, false);
    addBoxGeometry(cart, "tab", {0.12, 0.1, 0.06, 0.}, {0., y < 0. ? 0.15 : -0.15, 0.06}, color, false);
    root.addChild(raisim::Child(cart, raisim::Joint({1., 0., 0.}, {0., 0., 0.}, raisim::Mat<3, 3>::getIdentity(),
                                                    {-0.2, 2.6}, raisim::Joint::PRISMATIC, "slide"), "cart"));
    auto* system = world.addArticulatedSystem(root);
    system->setName(name);
    return system;
  }

  void reset(raisim::World& world) {
    removeIfAlive(world, pair);
    auto* driven = carriage(world, "driven", -0.2, {1., 0.45, 0.08, 1.});
    auto* passive = carriage(world, "passive", 0.2, {0.12, 0.45, 0.88, 1.});
    raisim::JoinOptions options;
    options.childPrefix = "passive/";
    options.name = "docked_pair";
    auto* docked = world.joinObjects(driven, driven->getBodyIdx("cart"), passive, passive->getBodyIdx("cart"),
                                     makeJoint(raisim::Joint::FIXED, "dock", {0., 0.2, 0.06}), options);
    loopJoints = docked->getLoopJointNames();
    pair = docked;
  }

  void update(raisim::World& world, int step, bool checks) {
    auto* system = static_cast<raisim::ArticulatedSystem*>(pair);
    Eigen::VectorXd force = Eigen::VectorXd::Zero(system->getDOF());
    force[system->getMappingFromBodyIndexToGeneralizedVelocityIndex()[system->getBodyIdx("cart")]] =
        step < 3000 ? 0.3 : -0.3;
    system->setGeneralizedForce(force);
    const auto& velocity = system->getGeneralizedVelocity();
    if (checks && step > 0 && step < 3000)
      dockedError = std::max(dockedError, std::abs(velocity[0] - velocity[1]));
    if (step == 3000) pair = world.detachJoint(system, "dock").front();
  }

  void check(raisim::World&) const {
    auto* system = static_cast<raisim::ArticulatedSystem*>(pair);
    const auto& velocity = system->getGeneralizedVelocity();
    std::cout << "Docked velocity mismatch: " << dockedError << " m/s; undocked velocities: "
              << velocity.e().transpose() << " m/s" << std::endl;
    if (loopJoints != std::vector<std::string>{"dock"} || dockedError > 1e-8 ||
        !system->getLoopJointNames().empty() || velocity[0] > 0. || velocity[1] < 0.)
      throw std::runtime_error("the carriages did not move together while docked and separately afterwards");
  }
};
}  // namespace

int main(int argc, char** argv) {
  bool visualize = true;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--headless") visualize = false;
    else if (argument == "--activation-key" && i + 1 < argc) raisim::World::setActivationKey(argv[++i]);
    else throw std::invalid_argument("usage: articulated_join [--headless] [--activation-key FILE]");
  }
  raisim::World world;
  world.setTimeStep(0.001);
  world.setSleepingEnabled(false);
  world.addGround();
  ChainStation chain;
  GantryStation gantry;
  DockingStation docking;

  std::unique_ptr<raisim::RaisimServer> server;
  if (visualize) {
    server = std::make_unique<raisim::RaisimServer>(&world);
    server->launchServer();
    std::cout << "Connect rayrai_tcp_viewer to port " << server->getPort() << ". The six-second scene repeats.\n";
  }
  int step = 0;
  const auto beforeStep = [&] {
    const int phase = step % kCycle;
    if (phase == 0) {
      chain.reset(world);
      gantry.reset(world);
      docking.reset(world);
    }
    gantry.update(world, phase, !visualize);
    docking.update(world, phase, !visualize);
    ++step;
  };
  try {
    while (server ? !server->isTerminateRequested() : step < kCycle) {
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
  if (server) {
    server->killServer();
    return 0;
  }
  chain.check(world);
  gantry.check(world);
  docking.check(world);
  return 0;
}
