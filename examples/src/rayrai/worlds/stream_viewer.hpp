#pragma once
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rayrai/Material.hpp"
#include "rayrai/Visuals.hpp"
#include "rayrai/example_common.hpp"
#include "stream_scene.hpp"

// The creek world and its viewer, shared by the example and its GPU test. Wooden balls float down
// the creek: every physics step applies the buoyancy and drag of the simulated water to them.
struct StreamViewer {
  using Viewer = raisin::RayraiWindow;
  static constexpr int kBalls = 5;
  static constexpr double kBallRadius = .12, kBallDensity = 650, kTimeStep = .002;
  std::shared_ptr<raisim::World> world = std::make_shared<raisim::World>();
  std::shared_ptr<Viewer> viewer;
  std::vector<std::shared_ptr<raisin::InstancedVisuals>> foliage;
  std::vector<raisim::Sphere*> balls;
  std::vector<double> launches;
  double time = 0;
  // Time the physics still has to catch up with, less than one step.
  double physicsLag = 0;

  // rsc: the examples' resource directory; the plants and rocks are in rsc/forest.
  StreamViewer(const std::filesystem::path& rsc, int width, int height) {
    const auto assets = rsc / "forest";
    const char* names[stream::kPlantTypes] = {"tree_small_02",   "grass_bermuda_01", "grass_medium_01", "grass_medium_02",
                                              "fern_02",         "dandelion_01",     "nettle_plant",    "periwinkle_plant"};
    std::vector<std::filesystem::path> files{assets / "ground/mud_forest_diff_1k.jpg", assets / "ground/mud_forest_nor_gl_1k.jpg"};
    for (const char* name : names) files.push_back(assets / name / "model.gltf");
    for (int type = 0; type < 6; ++type) files.push_back(assets / ("rock_moss_" + std::to_string(type)) / "model.gltf");
    for (const auto& file : files)
      if (!std::filesystem::exists(file)) throw std::runtime_error("Missing creek asset: " + file.string());
    world->setTimeStep(kTimeStep);
    stream::addTerrain(*world);
    const double mass = kBallDensity * 4 / 3 * stream::kPi * kBallRadius * kBallRadius * kBallRadius;
    for (int i = 0; i < kBalls; ++i) {
      auto* ball = world->addSphere(kBallRadius, mass);
      ball->setName("ball " + std::to_string(i));
      ball->setAppearance("0.3, 0.17, 0.08, 1");
      balls.push_back(ball);
      launches.push_back(-1);
      park(i);
    }

    viewer = std::make_shared<Viewer>(world, width, height);
    viewer->setAsyncMeshLoadingEnabled(true);
    auto quality = Viewer::defaultRenderQualitySettings(Viewer::RenderQualityPreset::High);
    quality.gamma = 2.2f;
    quality.colorMode = raisin::ViewerColorMode::AcesApprox;
    quality.pbrExposure = 1.0f;
    quality.depthOfFieldEnabled = false;
    quality.reflectiveGround = false;
    quality.viewerMsaaSamples = 4;
    quality.directionalShadowCascadeCount = 3;
    quality.directionalShadowCascadeMaxDistance = 60;
    quality.addViewerFillLights = false;
    viewer->setRenderQualitySettings(quality);

    const auto plants = stream::scatterPlants();
    for (int type = 0; type < stream::kPlantTypes; ++type) {
      auto visual = viewer->addInstancedVisuals(names[type], raisim::Shape::Mesh, glm::vec3(1), glm::vec4(1),
                                                glm::vec4(.85f, .92f, .75f, 1), (assets / names[type] / "model.gltf").string(), true);
      std::vector<raisin::InstancedVisuals::InstanceSpec> instances;
      for (const auto& p : plants)
        if (p.type == type) instances.push_back(instance(p, type == stream::kTree ? 0 : .4f));
      visual->addInstances(instances);
      visual->setAutomaticMeshLodEnabled(true);
      visual->setProjectedLodPolicy(true, 2, 8);
      visual->setShadowFoliageLodPolicy(true, 2.5f, 16);
      visual->setSortTransparentInstances(false);
      visual->configureFoliageWind(0, type == stream::kTree ? 1.3f : .5f, type == stream::kTree ? .025f : .09f, .7f, .2f);
      visual->setFoliageImpostorPolicy(false);
      foliage.push_back(visual);
    }
    const auto rocks = stream::scatterRocks();
    stream::addCreekRockColliders(*world, assets);
    for (int type = 0; type < 6; ++type) {
      const std::string name = "rock_moss_" + std::to_string(type);
      auto visual = viewer->addInstancedVisuals(name, raisim::Shape::Mesh, glm::vec3(1), glm::vec4(1), glm::vec4(1),
                                                (assets / name / "model.gltf").string(), true);
      std::vector<raisin::InstancedVisuals::InstanceSpec> instances;
      for (const auto& p : rocks)
        if (p.type == type) instances.push_back(instance(p, 0));
      visual->addInstances(instances);
      visual->setAutomaticMeshLodEnabled(true);
      foliage.push_back(visual);
    }
    viewer->setWaterBody("creek", stream::waterBody());

    Viewer::WeatherSettings weather;
    weather.enabled = true;
    weather.preset = Viewer::WeatherPreset::Clear;
    weather.useExplicitSunAngles = true;
    weather.sunAzimuthDegrees = static_cast<float>(stream::kSunAzimuthDegrees);
    weather.sunElevationDegrees = static_cast<float>(stream::kSunElevationDegrees);
    weather.sunDiskSize = .001f;
    weather.windSpeed = 1.5f;
    weather.cloudCoverage = .3f;
    weather.cloudDensity = .8f;
    weather.cloudScale = .5f;
    weather.cloudAltitudeMeters = 1500;
    weather.cloudThicknessMeters = 700;
    weather.visibilityMeters = 6000;
    viewer->setWeatherSettings(weather);
    auto& sun = viewer->getLight();
    sun.type = raisin::LightType::DIRECTIONAL;
    sun.setShadowsEnabled(true);
    sun.setShadowResolution(2048);
    sun.setShadowParams(.0008f, 1.f, 1.25f);
    viewer->setShadowOrtho(45, .1f, 160);
    const double az = stream::kSunAzimuthDegrees * stream::kPi / 180, el = stream::kSunElevationDegrees * stream::kPi / 180;
    sun.direction = glm::vec3(std::cos(az) * std::cos(el), std::sin(az) * std::cos(el), -std::sin(el));
    sun.ambient = glm::vec3(.12f);
    sun.diffuse = glm::vec3(1, .88f, .72f) * 14.f;
    sun.specular = glm::vec3(14);
    viewer->setHeightmapPatternResourcePath((assets / "ground/mud_forest_diff_1k.jpg").string());
    viewer->setHeightmapNormalResourcePath((assets / "ground/mud_forest_nor_gl_1k.jpg").string());
    viewer->setHeightmapTerrainMaterialParameters(.45f, .98f, 0.f, false);

    auto& camera = viewer->getCamera();
    const auto eye = stream::cameraPosition(), target = stream::cameraTarget();
    const double dx = target[0] - eye[0], dy = target[1] - eye[1], dz = target[2] - eye[2];
    camera.position = {static_cast<float>(eye[0]), static_cast<float>(eye[1]), static_cast<float>(eye[2])};
    camera.yaw = static_cast<float>(std::atan2(dy, dx) * 180 / stream::kPi);
    camera.pitch = static_cast<float>(std::atan2(dz, std::hypot(dx, dy)) * 180 / stream::kPi);
    camera.zoom = 55;
  }

  // Run the water ahead of the view, the spring stronger at first, so the creek is running and the
  // pool full when the scene is shown. Renders a small camera that sees nothing but the sky.
  void warmUp(double seconds) {
    raisin::Camera sky;
    sky.ensureRenderTargets(16, 16);
    sky.position = glm::vec3(0, 0, 100);
    sky.front = glm::vec3(0, 0, 1);
    sky.up = sky.worldUp = glm::vec3(1, 0, 0);
    Viewer::RenderOverrides overrides;
    overrides.doShadows = false;
    overrides.drawVisualizationObjects = false;
    auto water = stream::waterBody();
    const float steady = water.flow.sources.front().flowRate;
    for (double t = 0; t < seconds; t += .1) {
      water.flow.sources.front().flowRate = t < .4 * seconds ? 6 * steady : steady;
      viewer->setWaterBody("creek", water);
      viewer->advanceWaterTime(.1);
      viewer->renderWithExternalCamera(sky, overrides);
    }
    viewer->setWaterBody("creek", stream::waterBody());
  }

  // Advance the balls and the water by dt: physics steps with the force of the water on each ball,
  // a new ball from the spring every few seconds and the ones that left the creek back to the spring.
  void step(double dt) {
    std::vector<raisin::WaterFlowSample> water(balls.size());
    for (size_t i = 0; i < balls.size(); ++i) {
      const auto p = balls[i]->getPosition();
      water[i] = viewer->sampleWaterFlow({p[0], p[1]});
    }
    for (physicsLag += dt; physicsLag >= kTimeStep - 1e-12; physicsLag -= kTimeStep) {
      for (size_t i = 0; i < balls.size(); ++i) {
        if (launches[i] < 0) continue;
        const auto p = balls[i]->getPosition(), v = balls[i]->getLinearVelocity(), w = balls[i]->getAngularVelocity();
        const auto force = stream::floatingForce({p[0], p[1], p[2]}, {v[0], v[1], v[2]}, kBallRadius, water[i]);
        balls[i]->setExternalForce(0, {force[0], force[1], force[2]});
        const double damping = water[i].wet ? .02 : 0;
        balls[i]->setExternalTorque(0, {-damping * w[0], -damping * w[1], -damping * w[2]});
      }
      world->integrate();
    }
    time += dt;
    for (size_t i = 0; i < balls.size(); ++i) {
      const auto p = balls[i]->getPosition();
      if (launches[i] < 0 && time >= 2 + 4.0 * static_cast<double>(i)) launch(i);
      if (launches[i] >= 0 && (p[0] > stream::kWaterHalfX - .3 || p[2] < -6 || time - launches[i] > 90)) launch(i);
    }
    viewer->advanceWaterTime(dt);
  }

 private:
  // Out of sight, away from the water.
  void park(size_t i) {
    balls[i]->setPosition(-19.5 + .4 * static_cast<double>(i), 13, stream::elevation(-19.5, 13) + 1);
    balls[i]->setBodyType(raisim::BodyType::STATIC);
  }

  void launch(size_t i) {
    const double x = stream::kSourceX + 1.2, y = stream::creekY(x) + .25 * std::sin(1.7 * static_cast<double>(i));
    balls[i]->setBodyType(raisim::BodyType::DYNAMIC);
    balls[i]->setPosition(x, y, stream::elevation(x, y) + .6);
    balls[i]->setVelocity(0, 0, 0, 0, 0, 0);
    launches[i] = time;
  }

  static raisin::InstancedVisuals::InstanceSpec instance(const stream::Placement& p, float colorWeight) {
    raisin::InstancedVisuals::InstanceSpec spec;
    spec.position = {p.x, p.y, p.z};
    // Rayrai's vec4 instance orientation is WXYZ.
    spec.orientation = {std::cos(p.yaw * .5), 0, 0, std::sin(p.yaw * .5)};
    spec.scale = glm::vec3(static_cast<float>(p.scale));
    spec.colorWeight = colorWeight * static_cast<float>(p.scale / std::max(1., p.scale));
    return spec;
  }
};
