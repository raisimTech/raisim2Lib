// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// A Robotiq 2F-85 gripper picks up a box, lifts it, and puts it back, over and over. The gripper
// has a single actuated joint (the left knuckle); its other five finger joints follow it through
// URDF <mimic> constraints, which keep the finger pads parallel just like the real linkage.

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"
#include "example_resources.hpp"

#include <Eigen/Core>

#include <cmath>

namespace {

constexpr double kCycle = 12.;       // [s] one pick-and-place cycle
constexpr double kHoverHeight = 0.35;  // [m] lift position with the pads clear of the box
constexpr double kGraspHeight = 0.17;  // [m] lift position with the pads around the box
constexpr double kCarryHeight = 0.40;  // [m]
constexpr double kClosed = 0.75;     // [rad] knuckle target when gripping; the box stops the
                                     // fingers earlier, so the PD squeezes it
constexpr double kBoxWidth = 0.04;   // [m] the open gripper spans 85 mm between its pads

/// smooth step from a (at t0) to b (at t1)
double ramp(double t, double t0, double t1, double a, double b) {
  if (t <= t0) return a;
  if (t >= t1) return b;
  const double s = (t - t0) / (t1 - t0);
  return a + (b - a) * s * s * (3. - 2. * s);
}

}  // namespace

int main(int argc, char* argv[]) {
  raisim::World::setActivationKey(exampleRscPath(argv[0], "activation.raisim"));

  raisim::World world;
  world.setTimeStep(0.001);
  // the targets below change every step; a sleeping gripper would not follow them
  world.setSleepingEnabled(false);
  auto* ground = world.addGround(0., "ground");
  ground->setAppearance("checkerboard");

  auto* gripper = world.addArticulatedSystem(
      exampleRscPath(argv[0], "robotiq_2f85/robotiq_2f85_mimic.urdf"));
  gripper->setName("robotiq_2f85");
  for (auto& body : gripper->getCollisionBodies()) body.setMaterial("pad");

  auto* box = world.addBox(kBoxWidth, kBoxWidth, 0.06, 0.1, "box");
  box->setName("box");
  box->setPosition(0., 0., 0.03);
  box->setAppearance("0.95,0.55,0.15,1.0");
  world.setMaterialPairProp("pad", "box", 1.0, 0.0, 0.001);
  world.setMaterialPairProp("ground", "box", 0.8, 0.0, 0.001);

  // only the lift and the left knuckle are driven; the mimic joints have no gains of their own
  const size_t lift = gripper->getGeneralizedVelocityIndex("lift");
  const size_t knuckle = gripper->getGeneralizedVelocityIndex("robotiq_85_left_knuckle_joint");
  Eigen::VectorXd gc = gripper->getGeneralizedCoordinate().e();
  gc[lift] = kHoverHeight;
  gripper->setGeneralizedCoordinate(gc);
  Eigen::VectorXd pGain = Eigen::VectorXd::Zero(gripper->getDOF());
  Eigen::VectorXd dGain = Eigen::VectorXd::Zero(gripper->getDOF());
  pGain[lift] = 5000.;
  dGain[lift] = 200.;
  pGain[knuckle] = 10.;
  dGain[knuckle] = 0.2;
  gripper->setPdGains(pGain, dGain);
  const Eigen::VectorXd zeroVelocity = Eigen::VectorXd::Zero(gripper->getDOF());

  raisim::RaisimServer server(&world);
  server.launchServer();
  raisim_examples::warnIfNoClientConnected(server);
  server.setCameraPositionAndLookAt({0.45, -0.45, 0.4}, {0., 0., 0.15});

  for (;;) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    // the world time stops while the viewer pauses the simulation, and so does the script
    const double t = std::fmod(world.getWorldTime(), kCycle);
    double height, grip;
    if (t < 3.) {  // open, descend around the box
      height = ramp(t, 1., 3., kHoverHeight, kGraspHeight);
      grip = 0.;
    } else if (t < 7.5) {  // close, lift, hold
      height = ramp(t, 4., 6., kGraspHeight, kCarryHeight);
      grip = ramp(t, 3., 4., 0., kClosed);
    } else if (t < 10.5) {  // put it down, open
      height = ramp(t, 7.5, 9.5, kCarryHeight, kGraspHeight);
      grip = ramp(t, 9.5, 10.5, kClosed, 0.);
    } else {  // rise
      height = ramp(t, 10.5, 12., kGraspHeight, kHoverHeight);
      grip = 0.;
    }
    Eigen::VectorXd target = Eigen::VectorXd::Zero(gripper->getDOF());
    target[lift] = height;
    target[knuckle] = grip;
    gripper->setPdTarget(target, zeroVelocity);
    server.integrateWorldThreadSafe();
  }
}
