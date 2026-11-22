#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <glm/gtc/quaternion.hpp>

#include "example_resources.hpp"
#include "raisim/World.hpp"
#include "rayrai/Visuals.hpp"
#include "rayrai/example_common.hpp"
#include "rayrai_example_compat.hpp"

int main(int argc, char* argv[]) {
  ExampleApp app;
  if (!app.init("rayrai_pbr_texture_maps", 1280, 720)) return -1;

  auto world = std::make_shared<raisim::World>();
  world->addGround();

  auto viewer = std::make_shared<raisin::RayraiWindow>(world, 1280, 720);
  auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(raisin::RayraiWindow::RenderQualityPreset::Ultra);
  quality.pbrExposure = 0.82f;
  quality.pbrEnvironmentIntensity = 0.28f;
  quality.pbrEnvironmentMaxLod = 5.0f;
  quality.pbrKeyLightIntensity = 1.0f;
  quality.addViewerFillLights = false;
  quality.gamma = 2.2f;
  quality.pbrToneMapping = true;
  quality.colorMode = raisin::RayraiWindow::ViewerColorMode::UnrealPreviewApprox;
  quality.bloomEnabled = true;
  quality.bloomThreshold = 1.45f;
  quality.bloomStrength = 0.08f;
  quality.screenSpaceAoEnabled = true;
  quality.screenSpaceAoRadius = 4.0f;
  quality.screenSpaceAoStrength = 0.42f;
  quality.mainLightAmbient = glm::vec3(0.34f);
  quality.mainLightDiffuse = glm::vec3(1.55f);
  quality.mainLightSpecular = glm::vec3(0.02f);
  quality.mainLightDirection = glm::normalize(glm::vec3(-0.35f, -0.55f, -0.78f));
  quality.fogDensity = 0.0f;
  quality.heightFogEnabled = false;
  quality.heightFogDensity = 0.0f;
  quality.volumetricFogEnabled = false;
  quality.volumetricFogDensity = 0.0f;
  quality.volumetricLightingEnabled = false;
  quality.reflectiveGround = false;
  quality.shadowStrength = 0.22f;
  quality.depthOfFieldEnabled = false;
  quality.shadowOrthoHalfSize = 7.0f;
  quality.shadowCenterOffset = 3.0f;
  viewer->setRenderQualitySettings(quality);
  raisim_examples::setRayraiBackgroundColorRgb255(*viewer, {38, 42, 52, 255});
  viewer->setFogDensity(0.0f);
  viewer->setShadowOrtho(7.0f, 0.1f, 30.0f);
  viewer->setShadowCenterOffset(3.0f);

  raisin::RayraiWindow::AdditionalLight warmFill;
  warmFill.type = raisin::LightType::POINT;
  warmFill.position = glm::vec3(2.4f, -3.0f, 3.2f);
  warmFill.diffuse = glm::vec3(1.35f, 1.12f, 0.90f);
  warmFill.specular = glm::vec3(0.0f);
  warmFill.linear = 0.09f;
  warmFill.quadratic = 0.032f;
  viewer->addAdditionalLight(warmFill);

  unsigned int environment = 0;
  unsigned int irradiance = 0;
  unsigned int prefiltered = 0;
  unsigned int brdf = 0;
  const std::string hdrPath = exampleRscPath(argv[0], "rayrai/hdr/polyhaven/potsdamer_platz_1k.hdr");
  if (std::filesystem::exists(hdrPath)) {
    environment = raisin::RayraiWindow::loadHdrEquirectangularCubemap(hdrPath.c_str(), 128, true);
    irradiance = raisin::RayraiWindow::createHdrIrradianceCubemap(hdrPath.c_str(), 32, 64);
    prefiltered = raisin::RayraiWindow::createHdrPrefilteredEnvironmentCubemap(hdrPath.c_str(), 128, 6, 128);
    brdf = raisin::RayraiWindow::createSplitSumBrdfLut(128, 128);
  }

  struct PbrAsset {
    std::string name;
    std::string path;
    double scale;
    double yaw;
    glm::vec3 localCenter;
  };
  const auto uprightYawQuaternion = [](double yaw) {
    constexpr double kSqrtHalf = 0.7071067811865476;
    const double halfYaw = 0.5 * yaw;
    const double cy = std::cos(halfYaw);
    const double sy = std::sin(halfYaw);
    return glm::quat(static_cast<float>(cy * kSqrtHalf), static_cast<float>(cy * kSqrtHalf), static_cast<float>(sy * kSqrtHalf), static_cast<float>(sy * kSqrtHalf));
  };

  // Khronos glTF Sample Assets. Together these exercise base color, normal,
  // emissive, metallic-roughness, roughness-only, and occlusion texture import.
  // Scales normalize each asset's longest authored glTF extent to roughly the
  // same preview size; localCenter keeps offset mesh origins centered in slots.
  // Only CC0 models: renders of this scene appear in the documentation and
  // videos, and DamagedHelmet (CC BY-NC) and AntiqueCamera (UX3D trademark)
  // are not licensed for that.
  const std::vector<PbrAsset> assets = {
      {"flight_helmet", "rayrai/pbr/FlightHelmet/glTF/FlightHelmet.gltf", 1.75, 0.10, {-0.0203510f, 0.3579895f, 0.0142417f}},
      {"sci_fi_helmet", "rayrai/pbr/SciFiHelmet/glTF/SciFiHelmet.gltf", 0.43, 3.24, {0.0000000f, 0.0000001f, -0.0000005f}},
      {"lantern", "rayrai/pbr/Lantern/glTF/Lantern.gltf", 0.049, 0.12, {3.8231542f, 13.0160300f, 0.0000000f}},
      {"boombox", "rayrai/pbr/BoomBox/glTF/BoomBox.gltf", 62.0, 3.04, {0.0000000f, 0.0000000f, 0.0000000f}},
      {"avocado", "rayrai/pbr/Avocado/glTF/Avocado.gltf", 19.9, 0.05, {0.0000000f, 0.0314002f, -0.0000000f}},
      {"water_bottle", "rayrai/pbr/WaterBottle/glTF/WaterBottle.gltf", 4.8, -0.06, {0.0000000f, 0.0000000f, 0.0000000f}},
  };

  const std::array<glm::vec3, 6> previewCenters = {{
      {-0.95f, 0.02f, 2.10f},
      {0.95f, 0.02f, 2.10f},
      {-2.85f, -0.02f, 0.70f},
      {-0.95f, -0.02f, 0.70f},
      {0.95f, -0.02f, 0.70f},
      {2.85f, -0.02f, 0.70f},
  }};

  const auto setAssetPose = [&](const std::shared_ptr<raisin::Visuals>& visual, const PbrAsset& asset, const glm::vec3& previewCenter, double yaw) {
    const glm::quat orientation = uprightYawQuaternion(yaw);
    visual->setOrientation(orientation.w, orientation.x, orientation.y, orientation.z);
    const glm::vec3 centerOffset = orientation * (asset.localCenter * static_cast<float>(asset.scale));
    visual->setPosition(previewCenter - centerOffset);
  };

  std::vector<std::shared_ptr<raisin::Visuals>> visuals;
  visuals.reserve(assets.size());
  for (size_t i = 0; i < assets.size(); ++i) {
    const auto& asset = assets[i];
    const std::string meshPath = exampleRscPath(argv[0], asset.path);
    auto visual = viewer->addVisualMesh(asset.name, meshPath, asset.scale, asset.scale, asset.scale, 1.0f, 1.0f, 1.0f, 1.0f);
    setAssetPose(visual, asset, previewCenters[i], asset.yaw);
    visual->setTwoSided(true);
    visual->setDetectable(true);
    if (environment != 0) {
      visual->setPbrEnvironment(environment, irradiance, prefiltered, brdf, 1.0f);
    }
    visuals.push_back(visual);
  }

  auto& camera = viewer->getCamera();
  camera.target = {0.0f, 0.00f, 1.22f};
  camera.position = {0.05f, -5.55f, 1.58f};
  camera.yaw = 89.5f;
  camera.pitch = -3.7f;
  camera.farPlane = camera.zFar = 5000.0f;
  camera.setCameraFixedTarget(true);
  camera.setCameraFixedDistance(true);

  while (!app.quit) {
    app.processEvents();
    if (app.quit) break;

    world->integrate();
    viewer->pollAsyncMeshLoads(8);

    const double t = world->getWorldTime();
    for (size_t i = 0; i < visuals.size(); ++i) {
      setAssetPose(visuals[i], assets[i], previewCenters[i], assets[i].yaw + 0.18 * t);
    }

    app.beginFrame();
    app.renderViewer(*viewer);
    if (environment != 0) {
      raisin::RayraiWindow::RenderOverrides overrides;
      overrides.environmentBackgroundMap = environment;
      overrides.environmentBackgroundExposure = 0.55f;
      viewer->renderWithExternalCamera(viewer->getCamera(), overrides);
    }
    app.endFrame();
  }

  viewer.reset();
  app.shutdown();
  return 0;
}
