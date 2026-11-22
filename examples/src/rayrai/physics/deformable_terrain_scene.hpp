// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "../worlds/offroad_buggy_vehicle.hpp"
#include "raisim/World.hpp"
#include "rayrai/RayraiWindow.hpp"

namespace deformable_terrain {

constexpr size_t kXSamples = 201;
constexpr size_t kYSamples = 76;
constexpr double kTimeStep = 0.001;
constexpr const char* kWheelNames[] = {"fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel"};

/// The off-road buggy on a 16 m by 6 m bed of deformable soil. It drives at about 0.8 m/s and brakes near the end.
struct Scene {
  explicit Scene(const std::string& buggyUrdf) {
    world->setTimeStep(kTimeStep);
    raisim::DeformableTerrain::Material material;
    material.bekkerKphi = 2.0e6;
    material.sinkageExponent = 1.1;
    material.elasticStiffness = 2.0e7;
    material.damping = 4.0e4;
    material.cohesion = 100.0;
    soil = world->addDeformableTerrain(kXSamples, kYSamples, 16.0, 6.0, 0.0, 0.0, std::vector<double>(kXSamples * kYSamples, 0.0), material);
    soil->setName("soil");
    soil->setAppearance("0.49 0.325 0.188 1");
    soil->setColor(std::vector<raisim::ColorRGB>(kXSamples * kYSamples, {125, 83, 48}));
    buggy = std::make_unique<offroad::Buggy>(*world, buggyUrdf);
    buggy->place(-5.5, 0.0, 0.0, [](double, double) { return 0.0; });
    car = buggy->system();
    car->setName("terrain_car");
  }

  /// Drive for `count` physics steps, then average the buggy's telemetry over them.
  void advance(int count) {
    for (int i = 0; i < count; ++i) {
      offroad::DriverInput input;
      if (car->getGeneralizedCoordinate()[0] < 4.5) {
        const double speed = car->getGeneralizedVelocity()[0];
        input.throttle = std::clamp(0.20 + 0.60 * (0.8 - speed), 0.0, 0.65);
        input.brake = std::clamp(0.50 * (speed - 0.8), 0.0, 1.0);
      } else {
        input.brake = 1.0;
      }
      buggy->applyControls(input, kTimeStep);
      world->integrate();
      buggy->accumulate(kTimeStep);
    }
    steps += count;
    buggy->updateTelemetry(count * kTimeStep);
  }

  /// Print the car, suspension and soil state; return whether the car and soil state are finite.
  bool report() const {
    bool finite = car->getGeneralizedCoordinate().e().allFinite() && car->getGeneralizedVelocity().e().allFinite();
    const auto& state = soil->getSoilState();
    for (const auto* values : {&state.sinkage, &state.plasticSinkage, &state.shear})
      finite = finite && std::all_of(values->begin(), values->end(), [](double value) { return std::isfinite(value); });
    std::printf("simulation: %d steps, time=%.6f s\n", steps, world->getWorldTime());
    std::printf("car: x=%.3f m, speed=%.3f m/s\n", car->getGeneralizedCoordinate()[0], car->getGeneralizedVelocity()[0]);
    for (size_t i = 0; i < std::size(kWheelNames); ++i) {
      const auto& corner = buggy->telemetry().corners[i];
      std::printf("%s: speed=%.3f rad/s, shock=%.5f m, load=%.1f N\n", kWheelNames[i], corner.wheelSpeed, corner.shockExtension, corner.wheelLoad);
    }
    std::printf("suspension: maximum loop error=%.8f m\n", buggy->telemetry().maxLoopError);
    const auto& stats = soil->getSoilStepStats();
    const double rut = *std::max_element(state.plasticSinkage.begin(), state.plasticSinkage.end());
    std::printf("mesh: %zu vertices; last step: %zu contact rows, %zu changed vertices; maximum permanent rut=%.4f m\n",
                kXSamples * kYSamples, stats.contacts, stats.updatedVertices, rut);
    return finite;
  }

  std::shared_ptr<raisim::World> world = std::make_shared<raisim::World>();
  raisim::DeformableTerrain* soil = nullptr;
  std::unique_ptr<offroad::Buggy> buggy;
  raisim::ArticulatedSystem* car = nullptr;
  int steps = 0;
};

inline void configureLighting(raisin::RayraiWindow& viewer) {
  auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(raisin::RayraiWindow::RenderQualityPreset::Balanced);
  quality.highFidelityPbr = true;
  quality.gamma = 2.2f;
  quality.colorMode = raisin::ViewerColorMode::AcesApprox;
  quality.pbrToneMapping = true;
  quality.pbrExposure = 1.1f;
  quality.pbrEnvironmentIntensity = 1.8f;
  quality.pbrEnvironmentLightingTint = glm::vec3(1.0f);
  quality.mainLightDirection = glm::normalize(glm::vec3(-0.45f, 0.60f, -0.50f));
  quality.mainLightDiffuse = glm::vec3(1.0f, 0.96f, 0.90f) * 3.0f;
  quality.mainLightSpecular = glm::vec3(3.0f);
  quality.mainLightAmbient = glm::vec3(0.55f, 0.60f, 0.68f);
  quality.reflectiveGround = false;
  quality.depthOfFieldEnabled = false;
  quality.proceduralSkyBackgroundEnabled = true;
  viewer.setRenderQualitySettings(quality);
  // Sky irradiance lights the metal wishbones and dampers under the bodywork.
  viewer.generateWeatherSkyEnvironment(128, 32, true);
}

}  // namespace deformable_terrain
