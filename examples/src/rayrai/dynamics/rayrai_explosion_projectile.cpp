// Projectiles and explosions in a few lines: a gun fires at a wall of crates, in turn a solid round
// that only knocks crates with its momentum and a shell that explodes where it hits, and then a charge
// on the ground by the wall goes off. The wall is built again before each. RaiSim flies the
// projectiles and pushes the crates with the blast waves; rayrai draws the fireballs, smoke, dust,
// tracers and muzzle flashes from the physics snapshot, with no more code.
#include <cmath>
#include <cstdio>
#include <string>
#include <memory>
#include <vector>

#include "raisim/World.hpp"
#include "rayrai/PhysicsSnapshot.hpp"
#include "rayrai/example_common.hpp"
#include "rayrai_example_compat.hpp"

namespace {

constexpr int kColumns = 6, kRows = 4;
constexpr double kCrate = 0.6;
constexpr double kWallX = 18.0;

// Stacks the crates into a wall again, at rest.
void buildWall(std::vector<raisim::Box*>& crates) {
  for (int row = 0; row < kRows; ++row)
    for (int column = 0; column < kColumns; ++column) {
      auto* crate = crates[size_t(row * kColumns + column)];
      crate->setPosition(kWallX, (column - 0.5 * (kColumns - 1)) * kCrate, 0.5 * kCrate + row * kCrate);
      crate->setOrientation(1.0, 0.0, 0.0, 0.0);
      crate->setVelocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
}

}  // namespace

int main() {
  ExampleApp app;
  if (!app.init("rayrai_explosion_projectile", 1280, 720)) return -1;

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(0.002);
  // Air slows the projectiles; its velocity is the wind that carries the smoke.
  auto air = raisim::FluidMedium::air();
  air.velocity = {2.0, 1.0, 0.0};
  world->setAmbientFluid(air);
  auto* ground = world->addGround();
  ground->setName("ground");
  ground->setAppearance("checkerboard");

  std::vector<raisim::Box*> crates;
  for (int i = 0; i < kColumns * kRows; ++i) {
    auto* crate = world->addBox(kCrate, kCrate, kCrate, 40.0);
    crate->setName("crate " + std::to_string(i));
    crate->setAppearance(i % 2 ? "0.55,0.38,0.2,1" : "0.62,0.45,0.25,1");
    crates.push_back(crate);
  }
  buildWall(crates);

  // The gun: a barrel on a block; rounds leave its muzzle toward the wall.
  auto* mount = world->addBox(1.2, 1.0, 0.8, 1.0);
  mount->setBodyType(raisim::BodyType::STATIC);
  mount->setPosition(0.0, 0.0, 0.4);
  mount->setAppearance("0.3,0.33,0.3,1");
  auto* barrel = world->addCylinder(0.08, 2.0, 1.0);
  barrel->setBodyType(raisim::BodyType::STATIC);
  barrel->setPosition(1.3, 0.0, 1.0);
  barrel->setOrientation(std::sqrt(0.5), 0.0, std::sqrt(0.5), 0.0);
  barrel->setAppearance("0.25,0.27,0.25,1");
  const raisim::Vec<3> muzzle{2.35, 0.0, 1.0};

  raisin::PhysicsSnapshotCapture snapshotCapture;
  auto viewer = std::make_shared<raisin::RayraiWindow>(std::make_shared<raisim::World>(), 1280, 720);
  viewer->setRenderQualitySettings(
    raisin::RayraiWindow::defaultRenderQualitySettings(raisin::RayraiWindow::RenderQualityPreset::High));
  raisim_examples::setRayraiBackgroundColorRgb255(*viewer, {150, 170, 190, 255});
  raisim_examples::addRayraiBasicSceneLights(*viewer);
  // Compiles the effects' shaders now, so the first explosion does not stall a frame.
  viewer->prepareExplosionEffects();

  auto& camera = viewer->getCamera();
  camera.target = {9.0f, 0.0f, 1.2f};
  camera.position = {-3.0f, -11.0f, 5.5f};
  camera.yaw = 42.5f;
  camera.pitch = -15.0f;
  camera.setCameraFixedTarget(true);
  camera.setCameraFixedDistance(true);

  size_t nextHit = 0;
  int step = 0;
  while (!app.quit) {
    app.processEvents();
    if (app.quit) break;

    // Every two seconds the wall is built again and the next round goes off.
    const int round = step / 1000, phase = step % 1000;
    if (phase == 0) {
      buildWall(crates);
      const int kind = round % 3;
      if (kind == 0) {
        // A solid 10 kg round: no charge, so it stops where it hits and hands its momentum over.
        raisim::ProjectileProperties solid;
        solid.mass = 10.0;
        solid.radius = 0.05;
        solid.dragCoefficient = 0.25;
        world->addProjectile(muzzle, {300.0, 0.0, 9.0}, solid);
      } else if (kind == 1) {
        // A 12 kg shell with 1.5 kg of TNT, which detonates where it hits.
        raisim::ProjectileProperties shell;
        shell.mass = 12.0;
        shell.radius = 0.06;
        shell.dragCoefficient = 0.25;
        shell.explosion.chargeMass = 1.5;
        world->addProjectile(muzzle, {250.0, 2.0, 12.0}, shell);
      } else {
        // 3 kg of TNT on the ground beside the wall.
        raisim::ExplosionProperties charge;
        charge.chargeMass = 3.0;
        charge.surfaceBurst = true;
        world->addExplosion({kWallX + 1.5, -1.8, 0.1}, charge);
      }
    }

    world->integrate();

    // The world lists the hits of recent steps, oldest first, with increasing identifiers.
    for (const auto& hit : world->getProjectileHits()) {
      if (hit.id < nextHit) continue;
      nextHit = hit.id + 1;
      std::printf("projectile %zu hit %s at %.0f m/s\n", hit.projectileId,
                  world->getObjList()[hit.objectIndex]->getName().c_str(), hit.velocity.norm());
    }

    // Draw every 8th step, 62.5 frames a second of simulated time.
    if (step % 8 == 0) {
      viewer->applyPhysicsSnapshot(snapshotCapture.capture(*world));
      app.beginFrame();
      app.renderViewer(*viewer);
      app.endFrame();
    }
    ++step;
  }

  viewer.reset();
  app.shutdown();
  return 0;
}
