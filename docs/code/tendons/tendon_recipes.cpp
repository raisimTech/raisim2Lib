#include <raisim/World.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
// #region common
using Path = raisim::Tendon::PathElement;

void advance(raisim::World& world, int steps) {
  for (int i = 0; i < steps; ++i) world.integrate();
}
// #endregion common

void near(double actual, double expected, double tolerance, const char* message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
    throw std::runtime_error(std::string(message) + ": got " + std::to_string(actual));
}

void maximumLength() {
// #region limit
  raisim::World world;
  world.setTimeStep(0.001);
  world.setGravity({0., 0., 0.});
  auto* load = world.addSphere(0.05, 1.0);
  load->setPosition(1., 0., 0.);
  load->setLinearVelocity({2., 0., 0.});

  raisim::Tendon::Properties p;
  p.upperLimit = 1.2;  // Maximum distance from the world origin, in metres.
  auto* cable = world.addSpatialTendon("stop", {
      Path::via({nullptr, 0, {0., 0., 0.}}), Path::via({load})}, p);

  advance(world, 1000);
  cable->updateGeometry();
  std::cout << "length=" << cable->getLength()
            << " speed=" << cable->getVelocity() << '\n';
// #endregion limit
  near(cable->getLength(), 1.2, 1e-6, "maximum length");
  near(cable->getVelocity(), 0., 1e-6, "load must stop at the bound");
}

void lockCurrentLength() {
// #region lock
  raisim::World world;
  world.setTimeStep(0.001);
  world.setGravity({0., 0., -9.81});
  auto* load = world.addSphere(0.05, 1.0);
  load->setPosition(0., 0., 0.5);
  auto* cable = world.addSpatialTendon("pendulum", {
      Path::via({nullptr, 0, {0., 0., 2.}}), Path::via({load})});

  cable->updateGeometry();  // Also do this after manually changing poses.
  const double lockedLength = cable->getLength();
  auto p = cable->getProperties();
  p.lowerLimit = lockedLength;
  p.upperLimit = lockedLength;
  p.limitCompliance = 0.;   // Hard bilateral length constraint.
  p.positionCorrection = 0.2;
  cable->setProperties(p);

  load->setLinearVelocity({0.4, 0., 0.});  // The load may swing around the anchor.
  advance(world, 2000);
  cable->updateGeometry();
  std::cout << "locked=" << lockedLength << " actual=" << cable->getLength()
            << " reaction=" << cable->getLimitForce() << '\n';
// #endregion lock
  near(cable->getLength(), lockedLength, 0.002, "pendulum length");
// #region release
  auto released = cable->getProperties();
  released.lowerLimit = -std::numeric_limits<double>::infinity();
  released.upperLimit = std::numeric_limits<double>::infinity();
  cable->setProperties(released); // Retain other properties and drive settings.
// #endregion release
  load->setLinearVelocity({0., 0., -0.2});
  advance(world, 100);
  cable->updateGeometry();
  if (cable->getLength() < lockedLength + 0.01)
    throw std::runtime_error("released cable must extend beyond the old lock");
}

void elasticSuspension() {
// #region spring
  raisim::World world;
  world.setTimeStep(0.001);
  world.setGravity({0., 0., -9.81});
  auto* load = world.addSphere(0.05, 1.0);
  load->setPosition(0., 0., 0.8);

  raisim::Tendon::Properties p;
  p.springLower = 0.;
  p.springUpper = 1.0;      // Spring is slack below 1 m.
  p.stiffness = 200.;      // N/m.
  p.damping = 12.;         // N s/m; remains active inside the slack interval.
  p.frictionLoss = 0.05;   // N of dry friction along the transmission.
  p.armature = 0.1;        // kg of added transmission inertia.
  p.upperLimit = 1.8;      // Independent safety bound.
  auto* cable = world.addSpatialTendon("spring", {
      Path::via({nullptr, 0, {0., 0., 2.}}), Path::via({load})}, p);

  advance(world, 4000);
  cable->updateGeometry();
  std::cout << "length=" << cable->getLength()
            << " tension=" << cable->getTension()
            << " spring_energy=" << cable->getPotentialEnergy() << '\n';
// #endregion spring
  near(cable->getLength(), 1. + 9.81 / 200., 0.001, "spring equilibrium");
  near(cable->getVelocity(), 0., 0.001, "suspension must settle");
}

void forceControl() {
// #region force
  raisim::World world;
  world.setTimeStep(0.001);
  world.setGravity({0., 0., 0.});
  auto* load = world.addSphere(0.05, 1.0);
  load->setPosition(2., 0., 0.);
  auto* cable = world.addSpatialTendon("motor", {
      Path::via({nullptr, 0, {0., 0., 0.}}), Path::via({load})});

  auto p = cable->getProperties();
  p.actuationLower = -4.;  // Up to 4 N pulling.
  p.actuationUpper = 0.;   // No pushing from the actuator.
  cable->setProperties(p);
  cable->setDrive(raisim::Tendon::Drive{}); // Clear any previous servo/filter.
  cable->setTension(5.);   // Requests F = -5 N; the actuator clamps this to -4 N.

  advance(world, 100);     // 0.1 s: a 1 kg load acquires -0.4 m/s.
  cable->updateGeometry();
  std::cout << "actuator_force=" << cable->getActuationForce()
            << " velocity_x=" << load->getLinearVelocity()[0] << '\n';
// #endregion force
  near(cable->getActuationForce(), -4., 1e-9, "actuator clamp");
  near(load->getLinearVelocity()[0], -0.4, 1e-9, "force-driven motion");
}

void positionServo() {
// #region servo
  raisim::World world;
  world.setTimeStep(0.001);
  world.setGravity({0., 0., -9.81});
  auto* load = world.addSphere(0.05, 1.0);
  load->setPosition(0., 0., 0.8);

  raisim::Tendon::Properties p;
  p.actuationLower = -30.;
  p.actuationUpper = 0.;
  auto* cable = world.addSpatialTendon("winch", {
      Path::via({nullptr, 0, {0., 0., 2.}}), Path::via({load})}, p);
  raisim::Tendon::Drive drive;
  drive.targetLength = 0.8;
  drive.targetVelocity = 0.;
  drive.positionGain = 200.;
  drive.velocityGain = 25.;
  drive.force = -9.81;      // Feedforward balances this 1 kg vertical load.
  drive.activationTime = 0.05; // Filters feedforward; gains remain implicit.
  cable->setDrive(drive);

  advance(world, 3000);
  cable->updateGeometry();
  std::cout << "target=" << drive.targetLength
            << " actual=" << cable->getLength() << '\n';
// #endregion servo
  near(cable->getLength(), 0.8, 0.001, "position servo tracking");
  near(cable->getActuationForce(), -9.81, 0.02, "gravity compensation");
}

void wrappingAndPulley() {
// #region wrap
  raisim::World world;
  world.setTimeStep(0.001);
  world.setGravity({0., 0., 0.});
  auto* left = world.addSphere(0.05, 1.0);
  auto* right = world.addSphere(0.05, 1.0);
  auto* output = world.addSphere(0.05, 1.0);
  left->setPosition(-0.5, 0., 1.);
  right->setPosition(0.5, 0., 1.);
  output->setPosition(3., 0., 1.);

  // This static body makes the guide visible. Its collision masks are zero.
  auto* wheel = world.addCylinder(0.5, 0.2, 1., "default", 0, 0);
  wheel->setBodyType(raisim::BodyType::STATIC);
  wheel->setPosition(0., 0., 2.);
  wheel->setOrientation(std::sqrt(0.5), -std::sqrt(0.5), 0., 0.);
  auto guide = Path::cylinder({wheel}, 0.5); // Local Z axis becomes world Y.
  guide.withSideSite({nullptr, 0, {0., 0., 3.}}); // Route over the wheel.

  raisim::Tendon::Properties p;
  p.width = 0.01;
  p.color = {1., 0.55, 0.05, 1.};
  auto* cable = world.addSpatialTendon("pulley", {
      Path::via({left}), guide, Path::via({right}),
      Path::pulley(2.),  // Begins an independent branch at half force.
      Path::via({nullptr, 0, {3., 0., 3.}}), Path::via({output})}, p);
  cable->setTension(4.);  // Main branch: 4 N; output branch: 2 N.
  world.integrate();
  cable->updateGeometry(true);
  std::cout << "left_vz=" << left->getLinearVelocity()[2]
            << " output_vz=" << output->getLinearVelocity()[2]
            << " drawing_segments=" << cable->getVisualSegments().size() << '\n';
// #endregion wrap
  near(left->getLinearVelocity()[2], 0.004, 1e-8, "main branch force");
  near(output->getLinearVelocity()[2], 0.002, 1e-8, "pulley force ratio");
  if (cable->getVisualSegments().size() < 5)
    throw std::runtime_error("wrapped cable must have a drawable arc");
}

// #region fixed_model
raisim::ArticulatedSystem* makeTwoSliders(raisim::World& world) {
  // Two independent 1 kg sliders on a fixed base; no external model file.
  raisim::Body body(1., raisim::Mat<3, 3>::getIdentity(), {0., 0., 0.});
  raisim::Joint base;
  base.type = raisim::Joint::FIXED;
  base.name = "base";
  raisim::Child root(body, base, "base");
  raisim::Joint x({1., 0., 0.}, {0., 0., 0.},
      raisim::Mat<3, 3>::getIdentity(), {-10., 10.}, raisim::Joint::PRISMATIC, "x");
  raisim::Joint y({0., 1., 0.}, {0., 0., 0.},
      raisim::Mat<3, 3>::getIdentity(), {-10., 10.}, raisim::Joint::PRISMATIC, "y");
  root.addChild(raisim::Child(body, x, "link_x"));
  root.addChild(raisim::Child(body, y, "link_y"));
  return world.addArticulatedSystem(root);
}
// #endregion fixed_model

void fixedTransmission() {
// #region fixed
  raisim::World world;
  world.setTimeStep(0.001);
  world.setGravity({0., 0., 0.});
  auto* robot = makeTwoSliders(world);
  robot->setGeneralizedCoordinate({0., 0.});
  auto* transmission = world.addFixedTendon("differential", {
      {robot, "x", 1.0}, {robot, "y", 0.5}});

  raisim::Tendon::Drive drive;
  drive.targetLength = 0.5; // x + 0.5*y = 0.5; individual joints remain free.
  drive.positionGain = 200.;
  drive.velocityGain = 20.;
  transmission->setDrive(drive);
  advance(world, 2000);
  transmission->updateGeometry();
  std::cout << "transmission=" << transmission->getLength()
            << " joints=" << robot->getGeneralizedCoordinate().e().transpose() << '\n';
// #endregion fixed
  near(transmission->getLength(), 0.5, 0.001, "weighted coordinate tracking");
  near(robot->getGeneralizedCoordinate()[0], 0.4, 0.001, "first slider motion");
  near(robot->getGeneralizedCoordinate()[1], 0.2, 0.001, "second slider motion");
}

void coupledCables() {
// #region coupling
  raisim::World world;
  world.setTimeStep(0.001);
  world.setGravity({0., 0., 0.});
  world.setContactSolverParam(1., 1., 1., 100, 1e-12);
  auto* firstLoad = world.addSphere(0.05, 1.0);
  auto* secondLoad = world.addSphere(0.05, 1.0);
  firstLoad->setPosition(1., 0., 0.);
  secondLoad->setPosition(1., 2., 0.);
  auto* a = world.addSpatialTendon("A", {
      Path::via({nullptr, 0, {0., 0., 0.}}), Path::via({firstLoad})});
  auto* b = world.addSpatialTendon("B", {
      Path::via({nullptr, 0, {0., 2., 0.}}), Path::via({secondLoad})});

  raisim::TendonCoupling::Properties relation;
  relation.coefficients = {0., -0.65, 0., 0., 0.};
  world.addTendonCoupling("ratio", b, a, relation); // delta B = -0.65 * delta A.
  raisim::Tendon::Drive drive;
  drive.targetLength = a->getReferenceLength() + 0.2;
  drive.positionGain = 100.;
  drive.velocityGain = 20.;
  a->setDrive(drive);
  advance(world, 3000);
  a->updateGeometry();
  b->updateGeometry();
  const double deltaA = a->getLength() - a->getReferenceLength();
  const double deltaB = b->getLength() - b->getReferenceLength();
  std::cout << "delta_A=" << deltaA << " delta_B=" << deltaB
            << " residual=" << deltaB + 0.65 * deltaA << '\n';
// #endregion coupling
  near(deltaA, 0.2, 0.001, "driven cable displacement");
  near(deltaB, -0.13, 0.001, "coupled cable displacement");
  near(deltaB + 0.65 * deltaA, 0., 1e-6, "coupling residual");
}

void exportAndInspect(const std::string& exportPath) {
// #region export
  raisim::World world;
  world.setTimeStep(0.001);
  world.setGravity({0., 0., -9.81});
  auto* load = world.addSphere(0.05, 1.0);
  load->setName("load");  // Native XML resolves attachments by object name.
  load->setPosition(0., 0., 0.8);
  raisim::Tendon::Properties p;
  p.upperLimit = 1.2;
  auto* cable = world.addSpatialTendon("saved_cable", {
      Path::via({nullptr, 0, {0., 0., 2.}}), Path::via({load})}, p);
  cable->setTension(2.);

  auto appearance = cable->getProperties();
  appearance.width = 0.012; // Visual radius, not collision thickness.
  appearance.color = {0.05, 0.9, 0.75, 1.};
  cable->setProperties(appearance);
  cable->updateGeometry(true);
  for (const auto& segment : cable->getVisualSegments())
    std::cout << "segment " << segment.start.e().transpose()
              << " -> " << segment.end.e().transpose() << '\n';

  world.exportToXml(exportPath); // An absolute output filename supplied by main.
  raisim::World restored(exportPath);
  auto* restoredCable = restored.getTendon("saved_cable");
  if (!restoredCable) throw std::runtime_error("missing tendon after reload");
  advance(restored, 100);
  restoredCable->updateGeometry();
  std::cout << "reloaded_length=" << restoredCable->getLength()
            << " reloaded_force=" << restoredCable->getActuationForce() << '\n';
// #endregion export
  near(restoredCable->getLength(), 1.2, 1e-6, "reloaded hard limit");
  near(restoredCable->getActuationForce(), -2., 1e-9, "reloaded drive");
  near(restoredCable->getProperties().width, 0.012, 1e-12, "reloaded appearance");
}

void straightConnections() {
  {
// #region straight_lock
    raisim::World world;
    world.setTimeStep(0.001);
    world.setGravity({0., 0., 0.});
    auto* anchor = world.addSphere(0.02, 1.0);
    anchor->setBodyType(raisim::BodyType::STATIC);
    anchor->setPosition(0., 0., 0.);
    auto* load = world.addSphere(0.05, 1.0);
    load->setPosition(1., 0., 0.);
    load->setLinearVelocity({1., 0., 0.});
    raisim::Tendon::Properties p;
    p.lowerLimit = p.upperLimit = 1.0;
    auto* tendon = world.addSpatialTendon("lock", {Path::via({anchor}), Path::via({load})}, p);
    advance(world, 100);
    tendon->updateGeometry();
    std::cout << "target=" << tendon->getProperties().upperLimit
              << " current=" << tendon->getLength() << '\n';
// #endregion straight_lock
    near(tendon->getLength(), 1., 0.002, "length lock length");
    near(load->getLinearVelocity()[0], 0., 0.01, "length lock stops outward motion");
  }
  {
// #region straight_spring
    raisim::World world;
    world.setTimeStep(0.001);
    world.setGravity({0., 0., 0.});
    auto* anchor = world.addSphere(0.02, 1.0);
    anchor->setBodyType(raisim::BodyType::STATIC);
    anchor->setPosition(0., 0., 0.);
    auto* load = world.addSphere(0.05, 1.0);
    load->setPosition(1.1, 0., 0.);
    raisim::Tendon::Properties p;
    p.springLower = 0.;
    p.springUpper = 1.;
    p.stiffness = 100.;
    auto* tendon = world.addSpatialTendon("spring", {Path::via({anchor}), Path::via({load})}, p);
    world.integrate(); // Implicit spring force is approximately 10 N initially.
    std::cout << "tension=" << tendon->getTension() << '\n';
// #endregion straight_spring
    near(load->getLinearVelocity()[0], -0.01 / 1.0001, 1e-9, "implicit spring motion");
  }
  {
// #region straight_force
    raisim::World world;
    world.setTimeStep(0.001);
    world.setGravity({0., 0., 0.});
    auto* anchor = world.addSphere(0.02, 1.0);
    anchor->setBodyType(raisim::BodyType::STATIC);
    anchor->setPosition(0., 0., 0.);
    auto* load = world.addSphere(0.05, 1.0);
    load->setPosition(1., 0., 0.);
    auto* tendon = world.addSpatialTendon("actuator", {Path::via({anchor}), Path::via({load})});
    tendon->setTension(4.); // Positive pulls the endpoints together.
    advance(world, 100);
    std::cout << "velocity_x=" << load->getLinearVelocity()[0] << '\n';
// #endregion straight_force
    near(load->getLinearVelocity()[0], -0.4, 1e-9, "tendon force");
  }
}
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || std::string(argv[1]) == "--help") {
      std::cout << "Usage: tendon_recipes CASE [ACTIVATION_KEY|-] [EXPORT_XML]\n"
                   "Cases: limit lock spring force servo wrap fixed coupling export straight\n"
                   "The export case requires an absolute EXPORT_XML filename.\n";
      return argc < 2 ? 1 : 0;
    }
    const std::string recipe = argv[1];
    if (argc > 2 && std::string(argv[2]) != "-")
      raisim::World::setActivationKey(argv[2]);
    if (recipe == "limit") maximumLength();
    else if (recipe == "lock") lockCurrentLength();
    else if (recipe == "spring") elasticSuspension();
    else if (recipe == "force") forceControl();
    else if (recipe == "servo") positionServo();
    else if (recipe == "wrap") wrappingAndPulley();
    else if (recipe == "fixed") fixedTransmission();
    else if (recipe == "coupling") coupledCables();
    else if (recipe == "straight") straightConnections();
    else if (recipe == "export") {
      if (argc < 4 || !std::filesystem::path(argv[3]).is_absolute())
        throw std::invalid_argument("export requires an absolute output filename");
      exportAndInspect(argv[3]);
    } else throw std::invalid_argument("unknown recipe: " + recipe);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
