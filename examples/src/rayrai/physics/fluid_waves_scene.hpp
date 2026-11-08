#pragma once
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "raisim/World.hpp"
#include "rayrai/Water.hpp"

// Floating bodies on the swell of the sea: the waves rayrai draws hold them up. The sea of the world
// (World::addFluidVolume()) samples the same Gerstner waves at the physics time, so every body rides
// the surface the renderer shows. Shared by the example and its CPU test.
namespace fluid_waves {

constexpr double kTimeStep = 0.002;
constexpr double kPi = 3.14159265358979323846;
constexpr double kWoodDensity = 600.0;

// A moderate swell from the southwest
inline raisin::WaterBody waterBody() {
  auto body = raisin::defaultWaterBody(raisin::WaterPreset::Ocean);
  raisin::WaterWaveSpectrum spectrum;
  spectrum.waveCount = 12;
  spectrum.seed = 7u;
  spectrum.windAngleRadians = 0.6f;
  spectrum.minWavelength = 1.2f;
  spectrum.maxWavelength = 16.0f;
  spectrum.minAmplitude = 0.01f;
  spectrum.maxAmplitude = 0.1f;
  body.waves = raisin::generateGerstnerWaves(spectrum);
  return body;
}

class Scene {
 public:
  Scene() : water_(waterBody()) {
    world_.setTimeStep(kTimeStep);
    world_.setAmbientFluid(raisim::FluidMedium::air());
    raisim::FluidVolume sea;
    sea.medium = raisim::FluidMedium::seawater();
    sea.surfaceHeight = 0.0;
    // the surface over a body, and the water rising and falling with it: the first and second time
    // derivatives of the surface height, by central differences
    sea.sampler = [this](const raisim::Vec<3>& point) {
      const double time = world_.getWorldTime(), delta = 1e-3;
      const auto surface = raisin::sampleWaterSurface(water_, {point[0], point[1]}, time);
      const double later = raisin::sampleWaterSurface(water_, {point[0], point[1]}, time + delta).height;
      const double earlier = raisin::sampleWaterSurface(water_, {point[0], point[1]}, time - delta).height;
      raisim::FluidSample sample;
      sample.surfaceHeight = surface.height;
      sample.surfaceNormal = {surface.normal.x, surface.normal.y, surface.normal.z};
      sample.velocity = {0.0, 0.0, (later - earlier) / (2.0 * delta)};
      sample.acceleration = {0.0, 0.0, (later - 2.0 * surface.height + earlier) / (delta * delta)};
      return sample;
    };
    world_.addFluidVolume(sea);

    // wooden crates in a row
    for (int i = 0; i < 4; ++i) {
      const double side = 0.5 + 0.1 * i;
      auto* crate = world_.addBox(side, side, side, kWoodDensity * side * side * side);
      crate->setName("crate_" + std::to_string(i));
      crate->setAppearance("0.55,0.36,0.18,1");
      crate->setPosition(-3.0 + 2.0 * i, -2.0, 0.3);
      crate->setOrientation(Eigen::Matrix3d(Eigen::AngleAxisd(0.4 * i, Eigen::Vector3d::UnitZ())));
      floats_.push_back(crate);
    }
    // a raft of logs
    auto* raft = world_.addBox(2.4, 1.4, 0.18, 450.0 * 2.4 * 1.4 * 0.18);
    raft->setName("raft");
    raft->setAppearance("0.62,0.47,0.3,1");
    raft->setPosition(0.0, 1.0, 0.2);
    floats_.push_back(raft);
    // barrels on their sides
    for (int i = 0; i < 3; ++i) {
      const double radius = 0.3, length = 0.9;
      auto* barrel = world_.addCylinder(radius, length, 500.0 * kPi * radius * radius * length);
      barrel->setName("barrel_" + std::to_string(i));
      barrel->setAppearance("0.15,0.35,0.6,1");
      barrel->setPosition(-2.5 + 2.5 * i, 3.5, 0.4);
      barrel->setOrientation(Eigen::Matrix3d(Eigen::AngleAxisd(0.5 * kPi, Eigen::Vector3d::UnitX()) *
                                             Eigen::AngleAxisd(0.7 * i, Eigen::Vector3d::UnitY())));
      floats_.push_back(barrel);
    }
    // beach balls, light and large
    for (int i = 0; i < 3; ++i) {
      auto* ball = world_.addSphere(0.3, 0.4);
      ball->setName("ball_" + std::to_string(i));
      ball->setAppearance(i == 0 ? "0.9,0.15,0.1,1" : i == 1 ? "0.95,0.8,0.1,1" : "0.1,0.6,0.25,1");
      ball->setPosition(3.0 + 0.8 * i, -0.5 + 1.2 * i, 0.5);
      floats_.push_back(ball);
    }
  }
  Scene(const Scene&) = delete;
  Scene& operator=(const Scene&) = delete;

  void Step() { world_.integrate(); }
  raisim::World& world() { return world_; }
  const raisin::WaterBody& water() const { return water_; }
  const std::vector<raisim::SingleBodyObject*>& floats() const { return floats_; }
  // the surface over a point at the present physics time
  double SurfaceAt(double x, double y) const {
    return raisin::sampleWaterSurface(water_, {x, y}, world_.getWorldTime()).height;
  }

 private:
  raisim::World world_;
  raisin::WaterBody water_;
  std::vector<raisim::SingleBodyObject*> floats_;
};

}  // namespace fluid_waves
