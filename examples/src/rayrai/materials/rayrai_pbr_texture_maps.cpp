#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <glbinding/gl/gl.h>
#include <glm/gtc/quaternion.hpp>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "example_resources.hpp"
#include "raisim/World.hpp"
#include "raisim/math.hpp"
#include "rayrai/Visuals.hpp"
#include "rayrai/example_common.hpp"
#include "rayrai_example_compat.hpp"
#include "rayrai_example_utils.hpp"

namespace {

int wrapPixelX(int x, int width) {
  x %= width;
  return x < 0 ? x + width : x;
}

float sampleHdrBilinear(const float* pixels, int width, int height, int channels, float u, float v, int channel) {
  u -= std::floor(u);
  v = std::clamp(v, 0.0f, 1.0f);

  const float x = u * static_cast<float>(width) - 0.5f;
  const float y = v * static_cast<float>(height) - 0.5f;
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const float tx = x - static_cast<float>(x0);
  const float ty = y - static_cast<float>(y0);

  const auto value = [&](int sx, int sy) {
    sx = wrapPixelX(sx, width);
    sy = std::clamp(sy, 0, height - 1);
    return pixels[(static_cast<size_t>(sy) * width + sx) * channels + channel];
  };

  const float a = value(x0, y0) * (1.0f - tx) + value(x1, y0) * tx;
  const float b = value(x0, y1) * (1.0f - tx) + value(x1, y1) * tx;
  return a * (1.0f - ty) + b * ty;
}

glm::vec3 equirectangularDirection(float u, float v) {
  constexpr float kPi = 3.14159265358979323846f;
  const float theta = u * 2.0f * kPi - kPi;
  const float phi = v * kPi;
  const float sinPhi = std::sin(phi);
  return {
      sinPhi * std::cos(theta),
      sinPhi * std::sin(theta),
      std::cos(phi),
  };
}

std::pair<float, float> directionToEquirectangularUv(const glm::vec3& direction) {
  constexpr float kPi = 3.14159265358979323846f;
  const glm::vec3 d = glm::normalize(direction);
  float u = (std::atan2(d.y, d.x) + kPi) / (2.0f * kPi);
  const float v = std::acos(std::clamp(d.z, -1.0f, 1.0f)) / kPi;
  u -= std::floor(u);
  return {u, v};
}

glm::vec3 inverseRotate(const raisim::Mat<3, 3>& rotation, const glm::vec3& direction) {
  return {
      static_cast<float>(rotation[0] * direction.x + rotation[3] * direction.y + rotation[6] * direction.z),
      static_cast<float>(rotation[1] * direction.x + rotation[4] * direction.y + rotation[7] * direction.z),
      static_cast<float>(rotation[2] * direction.x + rotation[5] * direction.y + rotation[8] * direction.z),
  };
}

std::string rotateHdrEnvironment(const std::string& hdrPath, const raisim::Mat<3, 3>& rotation, const std::string& cacheName = "rayrai_pbr_texture_maps_hdr_rotated.hdr") {
  int width = 0;
  int height = 0;
  int channels = 0;
  float* pixels = stbi_loadf(hdrPath.c_str(), &width, &height, &channels, 3);
  if (!pixels || width <= 0 || height <= 0) {
    if (pixels) stbi_image_free(pixels);
    return hdrPath;
  }

  constexpr int kChannels = 3;
  std::vector<float> rotated(static_cast<size_t>(width) * height * kChannels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
      const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
      const glm::vec3 dstDirection = equirectangularDirection(u, v);
      const auto [srcU, srcV] = directionToEquirectangularUv(inverseRotate(rotation, dstDirection));
      const size_t dst = static_cast<size_t>(y * width + x) * kChannels;
      for (int channel = 0; channel < kChannels; ++channel) {
        rotated[dst + channel] = sampleHdrBilinear(pixels, width, height, kChannels, srcU, srcV, channel);
      }
    }
  }
  stbi_image_free(pixels);

  const auto rotatedPath = std::filesystem::temp_directory_path() / cacheName;
  if (!stbi_write_hdr(rotatedPath.string().c_str(), width, height, kChannels, rotated.data())) {
    std::cerr << "Failed to write rotated HDR environment: " << rotatedPath << std::endl;
    return hdrPath;
  }
  return rotatedPath.string();
}

std::string rotateHdrEnvironment(const std::string& hdrPath, const raisim::Vec<4>& quaternionWxyz, const std::string& cacheName = "rayrai_pbr_texture_maps_hdr_rotated.hdr") {
  raisim::Vec<4> unitQuaternion = quaternionWxyz;
  const double norm = std::sqrt(unitQuaternion[0] * unitQuaternion[0] + unitQuaternion[1] * unitQuaternion[1] + unitQuaternion[2] * unitQuaternion[2] + unitQuaternion[3] * unitQuaternion[3]);
  if (norm > 1.0e-12) {
    for (size_t i = 0; i < 4; ++i) unitQuaternion[i] /= norm;
  }

  raisim::Mat<3, 3> rotation;
  raisim::quatToRotMat(unitQuaternion, rotation);
  return rotateHdrEnvironment(hdrPath, rotation, cacheName);
}

}  // namespace

int main(int argc, char* argv[]) {
  std::filesystem::path screenshotPath;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--screenshot" && i + 1 < argc) {
      screenshotPath = argv[++i];
    }
  }

  ExampleApp app;
  const bool showWindow = screenshotPath.empty();
  if (!app.init("rayrai_pbr_texture_maps", 1280, 720, showWindow)) return -1;

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

  auto renderOneFrame = [&]() {
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
  };

  if (!screenshotPath.empty()) {
    while (viewer->pendingAsyncMeshLoadCount() != 0) {
      viewer->pollAsyncMeshLoads(std::numeric_limits<size_t>::max());
    }
    for (int i = 0; i < 3 && !app.quit; ++i) {
      renderOneFrame();
    }
    raisim_examples::saveCameraPng(camera, screenshotPath);
    std::cout << "Saved screenshot: " << screenshotPath.string() << "\n";
    viewer.reset();
    app.shutdown();
    return 0;
  }

  while (!app.quit) {
    app.processEvents();
    if (app.quit) break;

    renderOneFrame();
  }

  viewer.reset();
  app.shutdown();
  return 0;
}
