#pragma once
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pond_scene.hpp"
#include "rayrai/Material.hpp"
#include "rayrai/Visuals.hpp"
#include "rayrai/example_common.hpp"

// The pond world and its viewer, shared by the example and its GPU test.
struct PondViewer {
  using Viewer = raisin::RayraiWindow;
  std::shared_ptr<raisim::World> world = std::make_shared<raisim::World>();
  std::shared_ptr<Viewer> viewer;
  std::vector<std::shared_ptr<raisin::InstancedVisuals>> foliage;
  std::vector<std::shared_ptr<raisin::Visuals>> jetty;

  // rsc: the examples' resource directory; the plants and rocks are in rsc/forest.
  PondViewer(const std::filesystem::path& rsc, int width, int height) {
    const auto assets = rsc / "forest";
    const char* names[pond::kPlantTypes] = {"tree_small_02",   "grass_bermuda_01", "grass_medium_01", "grass_medium_02",
                                            "fern_02",         "dandelion_01",     "nettle_plant",    "periwinkle_plant"};
    const auto timber = rsc / "city/props/modular_street_seating/textures/modular_street_seating_timber";
    std::vector<std::filesystem::path> files{assets / "ground/mud_forest_diff_1k.jpg", assets / "ground/mud_forest_nor_gl_1k.jpg"};
    for (const char* name : names) files.push_back(assets / name / "model.gltf");
    for (int type = 0; type < 6; ++type) files.push_back(assets / ("rock_moss_" + std::to_string(type)) / "model.gltf");
    for (const char* map : {"_diff_1k.jpg", "_nor_gl_1k.jpg", "_arm_1k.jpg"}) files.push_back(timber.string() + map);
    for (const auto& file : files)
      if (!std::filesystem::exists(file)) throw std::runtime_error("Missing pond asset: " + file.string());
    pond::addTerrain(*world);

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
    quality.directionalShadowCascadeMaxDistance = 80;
    quality.addViewerFillLights = false;
    viewer->setRenderQualitySettings(quality);

    const auto plants = pond::scatterPlants();
    for (int type = 0; type < pond::kPlantTypes; ++type) {
      auto visual = viewer->addInstancedVisuals(names[type], raisim::Shape::Mesh, glm::vec3(1), glm::vec4(1),
                                                glm::vec4(.85f, .92f, .75f, 1), (assets / names[type] / "model.gltf").string(), true);
      std::vector<raisin::InstancedVisuals::InstanceSpec> instances;
      for (const auto& p : plants)
        if (p.type == type) instances.push_back(instance(p, type == pond::kTree ? 0 : .4f));
      visual->addInstances(instances);
      visual->setAutomaticMeshLodEnabled(true);
      visual->setProjectedLodPolicy(true, 2, 8);
      visual->setShadowFoliageLodPolicy(true, 2.5f, 16);
      visual->setSortTransparentInstances(false);
      visual->configureFoliageWind(0, type == pond::kTree ? 1.3f : .5f, type == pond::kTree ? .025f : .09f, .7f, .2f);
      visual->setFoliageImpostorPolicy(false);
      foliage.push_back(visual);
    }
    const auto rocks = pond::scatterRocks();
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

    addJetty(timber);
    viewer->setWaterBody("pond", pond::waterBody());

    Viewer::WeatherSettings weather;
    weather.enabled = true;
    weather.preset = Viewer::WeatherPreset::Clear;
    weather.useExplicitSunAngles = true;
    weather.sunAzimuthDegrees = static_cast<float>(pond::kSunAzimuthDegrees);
    weather.sunElevationDegrees = static_cast<float>(pond::kSunElevationDegrees);
    weather.sunDiskSize = .001f;
    weather.windSpeed = 1.5f;
    weather.cloudCoverage = .35f;
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
    viewer->setShadowOrtho(60, .1f, 180);
    const double az = pond::kSunAzimuthDegrees * pond::kPi / 180, el = pond::kSunElevationDegrees * pond::kPi / 180;
    sun.direction = glm::vec3(std::cos(az) * std::cos(el), std::sin(az) * std::cos(el), -std::sin(el));
    sun.ambient = glm::vec3(.12f);
    sun.diffuse = glm::vec3(1, .86f, .68f) * 14.f;
    sun.specular = glm::vec3(14);
    viewer->setHeightmapPatternResourcePath((assets / "ground/mud_forest_diff_1k.jpg").string());
    viewer->setHeightmapNormalResourcePath((assets / "ground/mud_forest_nor_gl_1k.jpg").string());
    viewer->setHeightmapTerrainMaterialParameters(.45f, .98f, 0.f, false);

    auto& camera = viewer->getCamera();
    const auto eye = pond::cameraPosition();
    camera.position = {static_cast<float>(eye[0]), static_cast<float>(eye[1]), static_cast<float>(eye[2])};
    camera.yaw = static_cast<float>(pond::kCameraYawDegrees);
    camera.pitch = static_cast<float>(pond::kCameraPitchDegrees);
    camera.zoom = static_cast<float>(pond::kCameraFovDegrees);
  }

  // Wooden boards, beams and posts with the timber texture, its grain along each part.
  void addJetty(const std::filesystem::path& timber) {
    raisin::Material wood = raisin::Material::pbr("jetty wood", glm::vec4(.7f, .66f, .62f, 1), 0, 1);
    wood.albedoMap = Viewer::loadTextureWithTiling((timber.string() + "_diff_1k.jpg").c_str(), true);
    wood.normalMap = Viewer::loadTextureWithTiling((timber.string() + "_nor_gl_1k.jpg").c_str(), false);
    wood.ormMap = Viewer::loadTextureWithTiling((timber.string() + "_arm_1k.jpg").c_str(), false);
    int index = 0;
    for (const auto& part : pond::jettyParts()) {
      const std::string name = "jetty " + std::to_string(index++);
      raisin::Material material = wood;
      std::shared_ptr<raisin::Visuals> visual;
      if (part.kind == pond::JettyPart::kPost) {
        visual = viewer->addVisualCylinder(name, part.size[0], part.size[2], glm::vec4(1));
        material.uvScaleOffset = {.25f, static_cast<float>(part.size[2] / 2), 0, 0};
        material.uvRotation = static_cast<float>(pond::kPi / 2);
      } else {
        visual = viewer->addVisualBox(name, part.size[0], part.size[1], part.size[2], glm::vec4(1));
        material.uvScaleOffset = {static_cast<float>(part.size[0] / 1.5), static_cast<float>(part.size[1]), 0, .13f * index};
      }
      visual->setPosition(part.position[0], part.position[1], part.position[2]);
      visual->setOrientation(std::cos(part.heading / 2), 0, 0, std::sin(part.heading / 2));
      visual->setMaterialOverride(material);
      visual->setDetectable(true);
      jetty.push_back(visual);
    }
  }

  static raisin::InstancedVisuals::InstanceSpec instance(const pond::Placement& p, float colorWeight) {
    raisin::InstancedVisuals::InstanceSpec spec;
    spec.position = {p.x, p.y, p.z};
    // Rayrai's vec4 instance orientation is WXYZ.
    spec.orientation = {std::cos(p.yaw * .5), 0, 0, std::sin(p.yaw * .5)};
    spec.scale = glm::vec3(static_cast<float>(p.scale));
    spec.colorWeight = colorWeight * static_cast<float>(p.scale / std::max(1., p.scale));
    return spec;
  }
};
