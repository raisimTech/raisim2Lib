// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// RaiSim without a renderer: a shell flies through the air into a stack of crates, explodes and
// throws them across the ground. The program prints where the crates come to rest.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "raisim/World.hpp"

namespace {

raisim::Vec<3> vec(double x, double y, double z) {
  raisim::Vec<3> v;
  v << x, y, z;
  return v;
}

}  // namespace

int main() {
  raisim::World world;
  world.setTimeStep(0.001);
  world.setAmbientFluid(raisim::FluidMedium::air());
  world.addGround();
  std::vector<raisim::Box*> crates;
  for (int layer = 0; layer < 4; ++layer)
    for (int k = 0; k < 4 - layer; ++k) {
      auto* crate = world.addBox(1.0, 1.0, 1.0, 100.0);
      crate->setPosition(12.0, -1.5 + 0.5 * layer + 1.05 * k, 0.5 + 1.0 * layer);
      crates.push_back(crate);
    }
  for (int i = 0; i < 500; ++i) world.integrate();

  raisim::ProjectileProperties shell;
  shell.mass = 10.0;
  shell.radius = 0.06;
  shell.dragCoefficient = 0.25;
  shell.explosion.chargeMass = 1.0;
  world.addProjectile(vec(0.0, 0.0, 1.2), vec(300.0, 0.0, 0.0), shell);
  size_t reported = 0;
  for (int i = 0; i < 5000; ++i) {
    world.integrate();
    for (const auto& explosion : world.getExplosions()) {
      if (explosion.id < reported) continue;
      reported = explosion.id + 1;
      std::printf("explosion %zu at (%.2f, %.2f, %.2f) m, t = %.3f s\n", explosion.id, explosion.position[0],
                  explosion.position[1], explosion.position[2], explosion.time);
    }
  }

  double farthest = 0.0;
  for (size_t i = 0; i < crates.size(); ++i) {
    const auto p = crates[i]->getPosition();
    const double thrown = std::hypot(p[0] - 12.0, p[1]);
    farthest = std::max(farthest, thrown);
    std::printf("crate %2zu rests at (%6.2f, %6.2f, %5.2f) m\n", i, p[0], p[1], p[2]);
  }
  std::printf("explosions %zu, farthest crate %.1f m from the stack\n", reported, farthest);
  return reported == 1 && farthest > 1.0 ? 0 : 1;
}
