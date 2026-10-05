// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// A 12-legged Strandbeest walks, driven by a single crank. Its 36 closed kinematic loops are
// closed by URDF <equality> constraints, which RaiSim eliminates exactly every step; the contact
// solver never iterates on them.

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"
#include "example_resources.hpp"

#include <Eigen/Core>

#include <cmath>
#include <iostream>

int main(int argc, char* argv[]) {
  raisim::World::setActivationKey(exampleRscPath(argv[0], "activation.raisim"));

  raisim::World world;
  world.setTimeStep(0.002);
  world.setSleepingEnabled(false);
  world.addGround(0., "ground")->setAppearance("checkerboard");

  auto* beest = world.addArticulatedSystem(exampleRscPath(argv[0], "strandbeest/strandbeest.urdf"));
  beest->setName("strandbeest");

  // count the loop constraints: each appears as one eliminated row of the contact problem
  world.integrate1();
  size_t loops = 0;
  for (const auto& row : *world.getContactProblem())
    loops += row.rank == raisim::contact::rank::PROJECTED_PIN_CONSTRAINT;
  world.integrate2();
  std::cout << "Strandbeest: " << beest->getDOF() << " degrees of freedom, " << loops
            << " loop constraints" << std::endl;

  // turn the crank at one revolution per second with a velocity-only PD; every other joint is
  // passive and moves through the loops
  const size_t crank = beest->getGeneralizedVelocityIndex("joint_crossbar_crank");
  Eigen::VectorXd dGain = Eigen::VectorXd::Zero(beest->getDOF());
  Eigen::VectorXd velocityTarget = Eigen::VectorXd::Zero(beest->getDOF());
  dGain[crank] = 200.;
  velocityTarget[crank] = 2. * M_PI;
  beest->setPdGains(Eigen::VectorXd::Zero(beest->getDOF()), dGain);
  beest->setPdTarget(beest->getGeneralizedCoordinate().e(), velocityTarget);

  raisim::RaisimServer server(&world);
  server.launchServer();
  raisim_examples::warnIfNoClientConnected(server);
  server.focusOn(beest);

  for (;;) {
    RS_TIMED_LOOP(int(world.getTimeStep() * 1e6))
    server.integrateWorldThreadSafe();
  }
}
