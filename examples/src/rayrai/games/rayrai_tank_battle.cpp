// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

// A tank on rolling hills fires shells at crates, walls, other tanks and barrels of fuel. Shells are
// World projectiles: they fly under gravity and the drag of the air, which the wind carries, hand their
// momentum to what they hit, and, when explosive, set off explosions whose blast waves push everything
// they reach and dig craters in the terrain. Barrels explode when hit or when a nearby blast is strong
// enough, so they set each other off.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <glbinding/gl/gl.h>
#include "example_resources.hpp"
#include "rayrai/example_common.hpp"
#include "rayrai_example_utils.hpp"

namespace {

using Viewer = raisin::RayraiWindow;
constexpr double kTimeStep = 0.002;
constexpr double kSprocketRadius = 0.35;
constexpr double kMuzzle = 3.2;
constexpr double kShellMass = 12.0;
constexpr double kReload = 1.2;
constexpr double kBarrelCharge = 4.0;
constexpr double kBarrelTriggerPressure = 1.5e5;

raisim::Vec<3> vec(double x, double y, double z) {
  raisim::Vec<3> v;
  v << x, y, z;
  return v;
}

raisim::Vec<3> vec(const Eigen::Vector3d& value) {
  raisim::Vec<3> v;
  v.e() = value;
  return v;
}

double terrainHeight(double x, double y) {
  const double hills = 2.2 * std::sin(0.045 * x) * std::cos(0.038 * y) + 0.9 * std::sin(0.11 * x + 0.7) * std::sin(0.09 * y + 0.3) +
                       0.25 * std::sin(0.31 * x) * std::cos(0.27 * y);
  const double flat = std::exp(-(x * x + y * y) / (2.0 * 18.0 * 18.0));
  return hills * (1.0 - flat);
}

struct Tank {
  raisim::ArticulatedSystem* system = nullptr;
  size_t left = 0, right = 0, turret = 0, gun = 0, turretCoordinate = 0, gunCoordinate = 0, gunBody = 0;
  Eigen::VectorXd pGain, dGain, pTarget, dTarget;
  double turretAim = 0.0, gunAim = -0.05;

  Tank(raisim::World& world, const std::string& urdf, const Eigen::Vector3d& position, double yaw) {
    system = world.addArticulatedSystem(urdf);
    left = system->getGeneralizedVelocityIndex("left_sprocket");
    right = system->getGeneralizedVelocityIndex("right_sprocket");
    turret = system->getGeneralizedVelocityIndex("turret");
    gun = system->getGeneralizedVelocityIndex("gun");
    turretCoordinate = system->getJoint("turret").getIdxInGeneralizedCoordinate();
    gunCoordinate = system->getJoint("gun").getIdxInGeneralizedCoordinate();
    gunBody = system->getBodyIdx("gun_link");
    Eigen::VectorXd gc = Eigen::VectorXd::Zero(system->getGeneralizedCoordinateDim());
    gc.head<3>() = position;
    gc[3] = std::cos(0.5 * yaw);
    gc[6] = std::sin(0.5 * yaw);
    system->setState(gc, Eigen::VectorXd::Zero(system->getDOF()));
    pGain = dGain = dTarget = Eigen::VectorXd::Zero(system->getDOF());
    pTarget = Eigen::VectorXd::Zero(system->getGeneralizedCoordinateDim());
    dGain[left] = dGain[right] = 4000.0;
    pGain[turret] = 4e5;
    dGain[turret] = 4e4;
    pGain[gun] = 1e6;
    dGain[gun] = 6e4;
    system->setPdGains(pGain, dGain);
    command(0.0, 0.0);
  }

  // Track speeds (m/s) for a throttle and a turn in [-1, 1].
  void command(double throttle, double turn) {
    const double speed = 9.0 * throttle, spin = 3.5 * turn;
    dTarget[left] = (speed - spin) / kSprocketRadius;
    dTarget[right] = (speed + spin) / kSprocketRadius;
    pTarget[turretCoordinate] = turretAim;
    pTarget[gunCoordinate] = gunAim;
    system->setPdTarget(pTarget, dTarget);
  }

  void muzzle(Eigen::Vector3d& position, Eigen::Vector3d& direction) const {
    raisim::Vec<3> p;
    raisim::Mat<3, 3> R;
    system->getLink("gun_link").getPose(p, R);
    direction = R.e().col(0);
    position = p.e() + direction * kMuzzle;
  }

  double heading() const {
    const auto& q = system->getGeneralizedCoordinate();
    return std::atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] * q[5] + q[6] * q[6]));
  }
};

// Turret and gun angles that point the gun at a target (straight, for a fast shell).
void aimAt(Tank& tank, const Eigen::Vector3d& target) {
  raisim::Vec<3> pivot;
  raisim::Mat<3, 3> R;
  tank.system->getLink("turret_link").getPose(pivot, R);
  const Eigen::Vector3d offset = target - pivot.e() - Eigen::Vector3d(0, 0, 0.3);
  tank.turretAim = std::remainder(std::atan2(offset.y(), offset.x()) - tank.heading(), 2 * M_PI);
  tank.gunAim = std::clamp(-std::atan2(offset.z(), offset.head<2>().norm()), -0.33, 0.13);
}

struct Barrel {
  raisim::Cylinder* body = nullptr;
  double fuse = -1.0;  // world time at which it goes off, once something set it off
};

void setOrbitDistance(raisin::Camera& camera, float distance) {
  const float yaw = glm::radians(camera.yaw), pitch = glm::radians(camera.pitch);
  const glm::vec3 front(std::cos(yaw) * std::cos(pitch), std::sin(yaw) * std::cos(pitch), std::sin(pitch));
  camera.setCameraFixedDistance(false);
  camera.position = camera.target - front * distance;
  camera.setCameraFixedDistance(true);
}

int run(int argc, char* argv[]) {
  std::string screenshot;
  int captureFrames = 240;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--screenshot" && i + 1 < argc) screenshot = argv[++i];
    else if (option == "--frames" && i + 1 < argc) captureFrames = std::stoi(argv[++i]);
    else {
      std::cerr << "Usage: rayrai_tank_battle [--screenshot PNG] [--frames N]\n";
      return 2;
    }
  }
  const bool capture = !screenshot.empty();

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(kTimeStep);
  // Sea-level air with a breeze: shells drift with it and smoke blows away downwind.
  raisim::FluidMedium air = raisim::FluidMedium::air();
  float windSpeed = 6.0f, windDirection = 90.0f;
  const auto setWind = [&]() {
    air.velocity = vec(windSpeed * std::cos(windDirection * M_PI / 180), windSpeed * std::sin(windDirection * M_PI / 180), 0.0);
    world->setAmbientFluid(air);
  };
  setWind();
  constexpr size_t kSamples = 241;
  constexpr double kExtent = 240.0;
  std::vector<double> heights(kSamples * kSamples);
  for (size_t j = 0; j < kSamples; ++j)
    for (size_t i = 0; i < kSamples; ++i)
      heights[j * kSamples + i] = terrainHeight(-0.5 * kExtent + kExtent * i / (kSamples - 1), -0.5 * kExtent + kExtent * j / (kSamples - 1));
  auto* terrain = world->addHeightMap(kSamples, kSamples, kExtent, kExtent, 0, 0, heights);
  terrain->setAppearance("0.42,0.45,0.30,1.0");

  const std::string urdf = exampleRscPath(argv[0], "tank/tank.urdf");
  Tank player(*world, urdf, {0.0, 0.0, 0.5}, 0.0);
  for (const auto& [x, y, yaw] : {std::array<double, 3>{52.0, 14.0, 2.6}, std::array<double, 3>{46.0, -20.0, 2.0}}) {
    Tank dummy(*world, urdf, {x, y, terrainHeight(x, y) + 0.6}, yaw);
    dummy.system->setName("target tank");
  }
  // Crate pyramids, concrete walls and fuel barrels down range.
  std::mt19937 random(7);
  for (const auto& [cx, cy] : {std::array<double, 2>{30.0, -6.0}, std::array<double, 2>{38.0, 9.0}, std::array<double, 2>{24.0, 15.0}}) {
    for (int layer = 0; layer < 3; ++layer)
      for (int k = 0; k < 3 - layer; ++k) {
        auto* crate = world->addBox(1.0, 1.0, 1.0, 150.0);
        crate->setPosition(cx, cy - 0.55 * (2 - layer) + 1.1 * k, terrainHeight(cx, cy) + 0.5 + layer * 1.0);
        crate->setAppearance("0.55,0.40,0.22,1.0");
      }
  }
  for (const auto& [cx, cy, yaw] : {std::array<double, 3>{20.0, -14.0, 0.3}, std::array<double, 3>{34.0, 0.5, 1.4}, std::array<double, 3>{44.0, -6.0, 0.0}}) {
    auto* wall = world->addBox(0.6, 6.0, 2.4, 20000.0);
    wall->setPosition(cx, cy, terrainHeight(cx, cy) + 1.0);
    wall->setOrientation(std::cos(0.5 * yaw), 0, 0, std::sin(0.5 * yaw));
    wall->setBodyType(raisim::BodyType::STATIC);
    wall->setAppearance("0.62,0.62,0.60,1.0");
  }
  std::vector<Barrel> barrels;
  for (const auto& [cx, cy] : {std::array<double, 2>{36.0, -12.0}, std::array<double, 2>{42.0, 4.0}, std::array<double, 2>{28.0, 4.0}}) {
    for (int k = 0; k < 4; ++k) {
      Barrel barrel;
      barrel.body = world->addCylinder(0.35, 1.0, 80.0);
      const double x = cx + 1.2 * (k % 2), y = cy + 1.2 * (k / 2);
      barrel.body->setPosition(x, y, terrainHeight(x, y) + 0.55);
      barrel.body->setAppearance("0.75,0.12,0.08,1.0");
      barrels.push_back(barrel);
    }
  }

  ExampleApp app;
  if (!app.init("RaiSim tank battle", 1600, 900, !capture)) return 1;
  if (capture) SDL_GL_SetSwapInterval(0);
  ImGui::GetIO().IniFilename = nullptr;
  int status = 0;
  {
    Viewer viewer(world, 1600, 900);
    viewer.setKeyboardInputEnabled(false);  // the keys drive the tank and aim the gun
    viewer.setRenderQualityPreset(raisin::RenderQualityPreset::High);
    viewer.prepareExplosionEffects();
    auto& camera = viewer.getCamera();
    camera.zoom = 55;
    camera.nearPlane = 0.1f;
    camera.target = {0.0f, 0.0f, 2.0f};
    camera.yaw = 0.0f;
    camera.pitch = -12.0f;
    camera.setCameraFixedTarget(true);
    setOrbitDistance(camera, 15.0f);

    double shellSpeed = 450.0, charge = 2.0, lastShot = -kReload;
    float throttleSlider = 0.0f, turnSlider = 0.0f;
    int shots = 0;
    bool showHelp = true, paused = false, explosive = true;
    auto last = std::chrono::steady_clock::now();
    size_t nextHit = 0, nextExplosion = 0;
    int frame = 0;

    const auto fire = [&]() {
      if (world->getWorldTime() - lastShot < kReload) return;
      Eigen::Vector3d position, direction;
      player.muzzle(position, direction);
      raisim::ProjectileProperties shell;
      shell.mass = kShellMass;
      shell.radius = 0.06;
      shell.dragCoefficient = 0.25;
      shell.maxLifetime = 12.0;
      shell.explosion.chargeMass = explosive ? charge : 0.0;
      shell.explosion.craterRadius = 0.9 * std::cbrt(charge);
      shell.explosion.craterDepth = 0.35 * std::cbrt(charge);
      shell.ignoredObject = player.system->getIndexInWorld();
      world->addProjectile(vec(position), vec(direction * shellSpeed), shell);
      // The gun recoils with the shell's momentum, for one step.
      player.system->setExternalForce(player.gunBody, vec(0.0, 0.0, 0.0), vec(-kShellMass * shellSpeed * direction / kTimeStep));
      lastShot = world->getWorldTime();
      ++shots;
    };

    while (!app.quit) {
      app.processEvents();
      if (app.quit) break;
      const auto now = std::chrono::steady_clock::now();
      const double frameTime = capture ? 1.0 / 60 : std::min(std::chrono::duration<double>(now - last).count(), 1.0 / 20);
      last = now;
      double throttle = throttleSlider, turn = turnSlider;
      bool trigger = false;
      if (!ImGui::GetIO().WantTextInput) {
        const Uint8* keys = SDL_GetKeyboardState(nullptr);
        throttle += (keys[SDL_SCANCODE_W] ? 1.0 : 0.0) - (keys[SDL_SCANCODE_S] ? 1.0 : 0.0);
        turn += (keys[SDL_SCANCODE_A] ? 1.0 : 0.0) - (keys[SDL_SCANCODE_D] ? 1.0 : 0.0);
        player.turretAim += frameTime * 0.9 * ((keys[SDL_SCANCODE_LEFT] ? 1.0 : 0.0) - (keys[SDL_SCANCODE_RIGHT] ? 1.0 : 0.0));
        player.gunAim = std::clamp(player.gunAim + frameTime * 0.3 * ((keys[SDL_SCANCODE_DOWN] ? 1.0 : 0.0) - (keys[SDL_SCANCODE_UP] ? 1.0 : 0.0)), -0.33, 0.13);
        trigger = keys[SDL_SCANCODE_SPACE];
        if (ImGui::IsKeyPressed(ImGuiKey_H, false)) showHelp = !showHelp;
        if (ImGui::IsKeyPressed(ImGuiKey_P, false)) paused = !paused;
      }
      const auto nearestBarrel = [&]() -> const Barrel* {
        const Barrel* best = nullptr;
        for (const auto& barrel : barrels)
          if (barrel.body && (!best || barrel.body->getPosition().norm() < best->body->getPosition().norm())) best = &barrel;
        return best;
      };
      if (capture) {
        // A scripted salvo at the nearest barrels.
        if (const Barrel* target = nearestBarrel()) aimAt(player, target->body->getPosition());
        trigger = frame % 70 == 40;
      }
      player.command(std::clamp(throttle, -1.0, 1.0), std::clamp(turn, -1.0, 1.0));
      if (trigger) fire();
      if (!paused) {
        const int steps = std::clamp(int(std::round(frameTime / kTimeStep)), 1, 25);
        for (int i = 0; i < steps; ++i) {
          world->integrate();
          // Barrels hit by a shell or caught in a strong enough blast go off a moment later.
          const auto& hits = world->getProjectileHits();
          for (const auto& hit : hits) {
            if (hit.id < nextHit) continue;
            nextHit = hit.id + 1;
            for (auto& barrel : barrels)
              if (barrel.body && barrel.fuse < 0 && barrel.body->getIndexInWorld() == hit.objectIndex) barrel.fuse = hit.time + 0.05;
          }
          const auto& explosions = world->getExplosions();
          for (const auto& explosion : explosions) {
            if (explosion.id < nextExplosion) continue;
            nextExplosion = explosion.id + 1;
            for (auto& barrel : barrels) {
              if (!barrel.body || barrel.fuse >= 0) continue;
              const double distance = (barrel.body->getPosition() - explosion.position.e()).norm();
              if (raisim::blast::wave(distance, explosion.properties.chargeMass).overpressure > kBarrelTriggerPressure)
                barrel.fuse = world->getWorldTime() + std::uniform_real_distribution<double>(0.08, 0.35)(random);
            }
          }
          for (auto& barrel : barrels) {
            if (!barrel.body || barrel.fuse < 0 || world->getWorldTime() < barrel.fuse) continue;
            const Eigen::Vector3d where = barrel.body->getPosition();
            world->removeObject(barrel.body);
            barrel.body = nullptr;
            raisim::ExplosionProperties blast;
            blast.chargeMass = kBarrelCharge;
            blast.surfaceBurst = true;
            blast.craterRadius = 1.6;
            blast.craterDepth = 0.6;
            world->addExplosion(vec(where), blast);
          }
        }
      }
      // Chase camera behind the turret.
      raisim::Vec<3> turretPosition;
      raisim::Mat<3, 3> turretRotation;
      player.system->getLink("turret_link").getPose(turretPosition, turretRotation);
      const glm::vec3 focus(turretPosition[0], turretPosition[1], turretPosition[2] + 1.6);
      camera.target = glm::mix(camera.target, focus, float(std::min(1.0, frameTime * 10)));
      const double aimHeading = std::atan2(turretRotation(1, 0), turretRotation(0, 0)) * 180.0 / M_PI;
      camera.yaw += float(std::remainder(aimHeading - camera.yaw, 360.0) * std::min(1.0, frameTime * 4.0));
      camera.pitch = -10.0f;
      setOrbitDistance(camera, 15.0f);

      app.beginFrame();
      app.renderViewer(viewer);
      if (!capture) {
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::Begin("Tank", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        const auto& v = player.system->getGeneralizedVelocity();
        ImGui::Text("Speed %.1f m/s   shots %d   reload %s", std::hypot(v[0], v[1]), shots,
                    world->getWorldTime() - lastShot < kReload ? "..." : "ready");
        int left = 0;
        for (const auto& barrel : barrels) left += barrel.body != nullptr;
        ImGui::Text("Fuel barrels left %d   shells in flight %zu", left, world->getProjectiles().size());
        if (ImGui::Button("Fire", ImVec2(120, 40))) fire();
        ImGui::SameLine();
        if (ImGui::Button("Aim at barrel", ImVec2(120, 40)))
          if (const Barrel* target = nearestBarrel()) aimAt(player, target->body->getPosition());
        ImGui::Checkbox("Explosive shells", &explosive);
        ImGui::SliderFloat("Throttle", &throttleSlider, -1.0f, 1.0f);
        ImGui::SliderFloat("Turn", &turnSlider, -1.0f, 1.0f);
        float turretDeg = float(player.turretAim * 180 / M_PI), gunDeg = float(-player.gunAim * 180 / M_PI);
        if (ImGui::SliderFloat("Turret (deg)", &turretDeg, -180.0f, 180.0f)) player.turretAim = turretDeg * M_PI / 180;
        if (ImGui::SliderFloat("Elevation (deg)", &gunDeg, -7.0f, 19.0f)) player.gunAim = -gunDeg * M_PI / 180;
        float speed = float(shellSpeed), tnt = float(charge);
        if (ImGui::SliderFloat("Shell speed (m/s)", &speed, 150.0f, 1200.0f)) shellSpeed = speed;
        if (ImGui::SliderFloat("Charge (kg TNT)", &tnt, 0.2f, 10.0f)) charge = tnt;
        if (ImGui::SliderFloat("Wind (m/s)", &windSpeed, 0.0f, 25.0f) | ImGui::SliderFloat("Wind from (deg)", &windDirection, -180.0f, 180.0f))
          setWind();
        if (showHelp)
          ImGui::TextUnformatted("W/S drive  A/D turn  arrows aim  Space fire\nH help  P pause  drag to look around");
        ImGui::End();
      }
      app.endFrame();
      ++frame;
      if (capture && frame == captureFrames) {
        status = raisim_examples::saveCameraPng(viewer.getCamera(), screenshot) ? 0 : 1;
        break;
      }
    }
  }  // Release graphics resources before closing the OpenGL context.
  app.shutdown();
  return status;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
