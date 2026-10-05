#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glbinding/gl/gl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "rayrai/example_common.hpp"
#include "rayrai/Visuals.hpp"
#include "example_resources.hpp"
#include "raisim/World.hpp"

namespace {

struct SceneDescription {
  std::filesystem::path meshPath;
  std::filesystem::path hdrPath;
  glm::vec3 cameraPosition{};
  glm::vec3 cameraTarget{};
  float verticalFov = 52.0f;
  float nearPlane = 0.05f;
  float farPlane = 50.0f;
  double timeStep = 0.002;
  int width = 1280;
  int height = 800;
};

bool loadSceneDescription(const std::filesystem::path& path,
                          SceneDescription& scene, std::string& error) {
  std::ifstream input(path);
  if (!input) {
    error = "Could not open scene: " + path.string();
    return false;
  }
  bool header = false, mesh = false, environment = false, camera = false;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    std::vector<std::string> tokens;
    for (std::string token; fields >> token;) tokens.push_back(token);
    if (tokens.empty() || tokens[0][0] == '#') continue;
    const auto& tag = tokens[0];
    try {
      if (tag == "raisim_engine_scene") {
        header = tokens.size() == 2 && tokens[1] == "1";
        if (!header) throw std::runtime_error("unsupported scene version");
      } else if (tag == "time_step") {
        if (tokens.size() != 2) throw std::runtime_error("invalid time_step");
        scene.timeStep = std::stod(tokens[1]);
      } else if (tag == "asset_root") {
        if (tokens.size() != 2 || tokens[1] != ".")
          throw std::runtime_error("expected a scene-local asset root");
      } else if (tag == "environment") {
        if (tokens.size() < 12 || environment)
          throw std::runtime_error("invalid environment record");
        scene.hdrPath = path.parent_path() / tokens[11];
        environment = true;
      } else if (tag == "object") {
        if (tokens.size() < 26 || mesh || tokens[2] != "mesh" ||
            tokens[22] != "visual_only")
          throw std::runtime_error("expected one visual-only mesh object");
        scene.meshPath = path.parent_path() / tokens[21];
        mesh = true;
      } else if (tag == "camera") {
        if (tokens.size() < 16 || camera)
          throw std::runtime_error("expected one camera");
        scene.cameraPosition = {std::stof(tokens[2]), std::stof(tokens[3]),
                                std::stof(tokens[4])};
        const glm::quat rotation(std::stof(tokens[5]), std::stof(tokens[6]),
                                 std::stof(tokens[7]), std::stof(tokens[8]));
        scene.cameraTarget = scene.cameraPosition +
                             glm::normalize(rotation) * glm::vec3(1.0f, 0.0f, 0.0f);
        scene.verticalFov = std::stof(tokens[9]);
        scene.nearPlane = std::stof(tokens[10]);
        scene.farPlane = std::stof(tokens[11]);
        scene.width = std::stoi(tokens[12]);
        scene.height = std::stoi(tokens[13]);
        camera = true;
      } else if (tag != "rayrai_render" && tag != "light") {
        throw std::runtime_error("unsupported record: " + tag);
      }
    } catch (const std::exception& exception) {
      error = path.string() + ": " + exception.what();
      return false;
    }
  }
  if (!header || !mesh || !environment || !camera || scene.width < 1 ||
      scene.height < 1 || scene.timeStep <= 0.0 || scene.verticalFov <= 0.0f) {
    error = "Blue Wall .rscene is missing a valid header, mesh, environment, or camera";
    return false;
  }
  return true;
}

void lookAt(raisin::Camera& camera, const glm::vec3& position,
            const glm::vec3& target, float verticalFov,
            float nearPlane, float farPlane) {
  camera.position = position;
  camera.target = target;
  camera.front = glm::normalize(target - position);
  camera.up = {0.0f, 0.0f, 1.0f};
  camera.worldUp = camera.up;
  camera.yaw = glm::degrees(std::atan2(camera.front.y, camera.front.x));
  camera.pitch = glm::degrees(std::asin(std::clamp(camera.front.z, -1.0f, 1.0f)));
  camera.zoom = verticalFov;
  camera.nearPlane = camera.zNear = nearPlane;
  camera.farPlane = camera.zFar = farPlane;
  camera.setCameraFixedTarget(false);
  camera.setCameraFixedDistance(false);
}

bool saveScreenshot(raisin::Camera& camera, const std::filesystem::path& path) {
  const int width = camera.rtWidth();
  const int height = camera.rtHeight();
  const size_t stride = static_cast<size_t>(width) * 4;
  std::vector<unsigned char> rgba(stride * static_cast<size_t>(height));
  std::vector<unsigned char> row(stride);
  gl::glBindTexture(gl::GL_TEXTURE_2D, camera.getFinalTexture());
  gl::glGetTexImage(gl::GL_TEXTURE_2D, 0, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
  gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
  for (int y = 0; y < height / 2; ++y) {
    auto* top = rgba.data() + static_cast<size_t>(y) * stride;
    auto* bottom = rgba.data() + static_cast<size_t>(height - 1 - y) * stride;
    std::memcpy(row.data(), top, stride);
    std::memcpy(top, bottom, stride);
    std::memcpy(bottom, row.data(), stride);
  }
  return stbi_write_png(path.string().c_str(), width, height, 4,
                        rgba.data(), static_cast<int>(stride)) != 0;
}

bool verifyCamera(raisin::Camera& camera) {
  const int width = camera.rtWidth();
  const int height = camera.rtHeight();
  std::vector<unsigned char> rgba(static_cast<size_t>(width) * height * 4);
  gl::glBindTexture(gl::GL_TEXTURE_2D, camera.getFinalTexture());
  gl::glGetTexImage(gl::GL_TEXTURE_2D, 0, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
  gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
  const auto pixel = [&](float x, float y) {
    const int column = static_cast<int>(x * width);
    const int row = height - 1 - static_cast<int>(y * height);
    return rgba.data() + (static_cast<size_t>(row) * width + column) * 4;
  };
  const auto* wall = pixel(0.85f, 0.15f);
  const auto* painting = pixel(0.60f, 0.22f);
  const auto* dresser = pixel(0.60f, 0.73f);
  const bool wallBlue = wall[2] > wall[0] * 1.3f && wall[2] > wall[1] * 1.1f;
  const bool paintingWarm = painting[0] > painting[1] * 1.15f &&
                            painting[0] > painting[2] * 1.2f;
  const bool dresserWood = dresser[0] > dresser[1] * 1.4f &&
                           dresser[0] > dresser[2] * 2.0f;
  std::cout << "Camera image check: blue wall=" << wallBlue
            << " painting=" << paintingWarm << " dresser=" << dresserWood << '\n';
  return wallBlue && paintingWarm && dresserWood;
}

bool compareReference(raisin::Camera& camera, const std::filesystem::path& path) {
  int width = 0, height = 0, channels = 0;
  unsigned char* reference = stbi_load(path.string().c_str(), &width, &height,
                                        &channels, 4);
  if (!reference || width != camera.rtWidth() || height != camera.rtHeight()) {
    std::cerr << "Could not load matching reference image: " << path << '\n';
    stbi_image_free(reference);
    return false;
  }
  std::vector<unsigned char> rgba(static_cast<size_t>(width) * height * 4);
  gl::glBindTexture(gl::GL_TEXTURE_2D, camera.getFinalTexture());
  gl::glGetTexImage(gl::GL_TEXTURE_2D, 0, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
  gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
  std::uint64_t totalError = 0;
  std::uint64_t changedPixels = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t rendered = (static_cast<size_t>(height - 1 - y) * width + x) * 4;
      const size_t expected = (static_cast<size_t>(y) * width + x) * 4;
      bool changed = false;
      for (int channel = 0; channel < 3; ++channel) {
        const int difference = std::abs(static_cast<int>(rgba[rendered + channel]) -
                                        static_cast<int>(reference[expected + channel]));
        totalError += static_cast<std::uint64_t>(difference);
        changed |= difference > 30;
      }
      changedPixels += changed;
    }
  }
  stbi_image_free(reference);
  const double meanError = static_cast<double>(totalError) / (width * height * 3);
  const double changedFraction = static_cast<double>(changedPixels) / (width * height);
  std::cout << "GLB reference comparison: mean_abs_rgb=" << meanError
            << " fraction_pixels_over_30=" << changedFraction << '\n';
  return meanError <= 8.0 && changedFraction <= 0.05;
}

} // namespace

int main(int argc, char* argv[]) {
  std::filesystem::path screenshot;
  std::filesystem::path recording;
  int recordingFrames = 60;  // Four seconds at 15 fps.
  std::filesystem::path reference;
  int benchmarkFrames = 0;
  bool checkCamera = false;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--screenshot" && i + 1 < argc) {
      screenshot = argv[++i];
    } else if (option == "--record" && i + 1 < argc) {
      recording = argv[++i];
    } else if (option == "--record-frames" && i + 1 < argc) {
      try { recordingFrames = std::stoi(argv[++i]); }
      catch (...) { return 2; }
      if (recordingFrames < 1) return 2;
    } else if (option == "--compare-reference" && i + 1 < argc) {
      reference = argv[++i];
    } else if (option == "--verify-camera") {
      checkCamera = true;
    } else if (option == "--benchmark-frames" && i + 1 < argc) {
      try {
        benchmarkFrames = std::stoi(argv[++i]);
      } catch (...) {
        std::cerr << "Invalid --benchmark-frames value\n";
        return 2;
      }
      if (benchmarkFrames < 1) return 2;
    } else {
      std::cerr << "Usage: rayrai_blue_wall_scene [--screenshot PNG] [--record FRAME_DIR] "
                   "[--record-frames N] "
                   "[--compare-reference PNG] [--verify-camera] "
                   "[--benchmark-frames N]\n";
      return 2;
    }
  }

  const std::filesystem::path scenePath =
      exampleRscPath(argv[0], "rayrai/blue_wall/blue_wall.rscene");
  SceneDescription description;
  std::string sceneError;
  if (!loadSceneDescription(scenePath, description, sceneError)) {
    std::cerr << sceneError << '\n';
    return 1;
  }
  if (!recording.empty()) {
    description.width = 640;
    description.height = 400;
    std::filesystem::create_directories(recording);
  }
  const std::filesystem::path lightsPath =
      description.meshPath.string() + ".rayrai_lights.json";
  for (const auto& path : {description.meshPath, lightsPath, description.hdrPath}) {
    if (!std::filesystem::is_regular_file(path)) {
      std::cerr << "Missing Blue Wall asset: " << path << '\n';
      return 1;
    }
  }

  ExampleApp app;
  if (!app.init("RayRai Blue Wall", description.width, description.height,
                screenshot.empty() && recording.empty() && reference.empty() && benchmarkFrames == 0 &&
                !checkCamera)) return 1;

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(description.timeStep);
  auto viewer = std::make_shared<raisin::RayraiWindow>(
      world, description.width, description.height);
  viewer->setAsyncMeshLoadingEnabled(false);
  auto settings = raisin::RayraiWindow::defaultRenderQualitySettings(
      raisin::RayraiWindow::RenderQualityPreset::Ultra);
  settings.backgroundColorRgb255 = {12.0f, 14.0f, 17.0f, 255.0f};
  settings.fogDensity = 0.0f;
  settings.gamma = 2.2f;
  settings.mainLightAmbient = glm::vec3(0.055f);
  settings.mainLightDiffuse = glm::vec3(0.42f);
  settings.mainLightSpecular = glm::vec3(0.08f);
  settings.mainLightDirection = glm::normalize(glm::vec3(-0.40f, -0.55f, -0.73f));
  settings.shadowStrength = 0.34f;
  settings.shadowBias = 0.00045f;
  settings.shadowPcfRadius = 1.6f;
  settings.reflectiveGround = false;
  settings.depthOfFieldEnabled = false;
  settings.highFidelityPbr = true;
  settings.pbrToneMapping = true;
  settings.colorMode = raisin::RayraiWindow::ViewerColorMode::UnrealPreviewApprox;
  settings.pbrExposure = 0.88f;
  settings.pbrEnvironmentIntensity = 0.18f;
  settings.pbrKeyLightIntensity = 1.25f;
  settings.addViewerFillLights = true;
  settings.bloomEnabled = true;
  settings.bloomThreshold = 1.35f;
  settings.bloomStrength = 0.10f;
  settings.bloomRadius = 5.5f;
  settings.screenSpaceAoEnabled = true;
  settings.screenSpaceAoRadius = 8.0f;
  settings.screenSpaceAoStrength = 0.36f;
  settings.screenSpaceAoBias = 0.016f;
  settings.shadowedLightBudget = 9;
  settings.maxPointShadowLights =
      static_cast<int>(raisin::RayraiWindow::kMaxAdditionalShadowLights);
  settings.additionalShadowResolutionScale = 0.5f;
  settings.pointShadowResolutionScale = 0.5f;
  settings.minAdditionalShadowResolution = 256;
  settings.updateShadowsEveryFrame = false;
  settings.autoSelectImportedShadowLight = true;
  if (!recording.empty()) {
    // Gallery captures do not need full-resolution interactive shadow maps.
    settings.shadowResolution = 2048;
    settings.pointShadowResolutionScale = 0.25f;
    settings.additionalShadowResolutionScale = 0.25f;
    settings.viewerMsaaSamples = 2;
  }
  viewer->setRenderQualitySettings(settings);

  raisin::RayraiWindow::SceneImportReport report;
  auto scene = viewer->importVisualScene("blue_wall", description.meshPath.string(),
                                         &report, true, 0.0022f);
  if (!scene) {
    std::cerr << "Could not import " << description.meshPath << '\n';
    viewer.reset();
    app.shutdown();
    return 1;
  }
  scene->setTwoSided(false);
  scene->setUseMeshColor(false);
  if (report.shadowLightPromoted ||
      viewer->promoteDominantAdditionalLightToMainShadowCaster(true, true)) {
    viewer->getLight().ambient = settings.mainLightAmbient;
    viewer->getLight().setShadowParams(0.0012f, 0.98f, 1.8f);
    viewer->setShadowOrtho(settings.shadowOrthoHalfSize, 0.05f, 18.0f);
  }

  const std::string hdr = description.hdrPath.string();
  const unsigned int environment =
      raisin::RayraiWindow::loadHdrEquirectangularCubemap(hdr.c_str(), 64, true);
  const unsigned int irradiance =
      raisin::RayraiWindow::createHdrIrradianceCubemap(hdr.c_str(), 16, 32);
  const unsigned int prefiltered =
      raisin::RayraiWindow::createHdrPrefilteredEnvironmentCubemap(hdr.c_str(), 64, 4, 32);
  const unsigned int brdf = raisin::RayraiWindow::createSplitSumBrdfLut(64, 64);
  if (!environment || !irradiance || !prefiltered || !brdf) {
    std::cerr << "Could not create Blue Wall HDR lighting\n";
    viewer.reset();
    app.shutdown();
    return 1;
  }
  scene->setPbrEnvironment(environment, irradiance, prefiltered, brdf, 0.24f);
  raisin::RayraiWindow::ReflectionProbeCaptureSettings probeSettings;
  probeSettings.resolution = 64;
  probeSettings.doShadows = false;
  probeSettings.drawCoordinateFrames = false;
  probeSettings.drawPointClouds = false;
  probeSettings.drawVisualizationObjects = true;
  probeSettings.includeEnvironmentBackground = true;
  probeSettings.environmentBackgroundMap = environment;
  probeSettings.environmentBackgroundExposure = 0.24f;
  const glm::vec3 probePosition{0.0f, -1.85f, 1.25f};
  const unsigned int roomMap =
      viewer->captureReflectionProbeCubemap(probePosition, probeSettings);
  if (roomMap) {
    raisin::RayraiWindow::ReflectionProbe probe;
    probe.position = probePosition;
    probe.radius = 4.0f;
    probe.environmentMap = environment;
    probe.irradianceMap = irradiance;
    probe.prefilteredEnvironmentMap = roomMap;
    probe.brdfLut = brdf;
    probe.strength = 0.24f;
    viewer->addReflectionProbe(probe);
    viewer->applyNearestReflectionProbe(*scene, probePosition);
  }

  auto& camera = viewer->getCamera();
  lookAt(camera, description.cameraPosition, description.cameraTarget,
         description.verticalFov, description.nearPlane, description.farPlane);
  std::cout << "Loaded Blue Wall .rscene: " << scenePath << '\n';
  std::cout << "Blue Wall camera: position=(" << camera.position.x << ','
            << camera.position.y << ',' << camera.position.z << ") vertical_fov="
            << camera.zoom << '\n';

  auto renderFrame = [&]() {
    world->integrate();
    app.processEvents();
    app.beginFrame();
    app.renderViewer(*viewer);
    app.endFrame();
  };
  if (!screenshot.empty() || !recording.empty() || !reference.empty() || benchmarkFrames > 0 || checkCamera) {
    for (int i = 0; i < 3; ++i) renderFrame();
    if (!recording.empty()) {
      const int frames = recordingFrames;
      // Orbit the room's focal point and return to the initial view for a smooth loop.
      const glm::vec3 pivot = description.cameraPosition +
          3.0f * glm::normalize(description.cameraTarget - description.cameraPosition);
      const glm::vec3 offset = description.cameraPosition - pivot;
      for (int frame = 0; frame < frames; ++frame) {
        const float angle = 0.065f * std::sin(2.0f * 3.14159265359f * frame / frames);
        const glm::vec3 eye = pivot + glm::vec3(
            std::cos(angle) * offset.x - std::sin(angle) * offset.y,
            std::sin(angle) * offset.x + std::cos(angle) * offset.y, offset.z);
        lookAt(camera, eye, pivot, description.verticalFov,
               description.nearPlane, description.farPlane);
        renderFrame();
        char name[32];
        std::snprintf(name, sizeof(name), "frame_%03d.png", frame);
        if (!saveScreenshot(camera, recording / name)) return 1;
        if ((frame + 1) % 25 == 0)
          std::cout << "Blue Wall: recorded " << frame + 1 << '/' << frames << std::endl;
      }
      lookAt(camera, description.cameraPosition, description.cameraTarget,
             description.verticalFov, description.nearPlane, description.farPlane);
      renderFrame();
    }
    if (benchmarkFrames > 0) {
      const auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < benchmarkFrames; ++i) renderFrame();
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
      std::cout << "Blue Wall benchmark: frames=" << benchmarkFrames
                << " total_ms=" << ms << " avg_ms=" << ms / benchmarkFrames << '\n';
    }
    if (!screenshot.empty() && !saveScreenshot(camera, screenshot)) {
      std::cerr << "Could not save " << screenshot << '\n';
      viewer.reset();
      app.shutdown();
      return 1;
    }
    if (checkCamera && !verifyCamera(camera)) {
      std::cerr << "Blue Wall camera framing check failed\n";
      viewer.reset();
      app.shutdown();
      return 1;
    }
    if (!reference.empty() && !compareReference(camera, reference)) {
      std::cerr << "Blue Wall quality differs from the GLB reference\n";
      viewer.reset();
      app.shutdown();
      return 1;
    }
  } else {
    while (!app.quit) renderFrame();
  }
  if (!recording.empty()) {
    // The shared GL asset cache outlives ExampleApp's context during normal teardown.
    // Recording processes exit directly, as the documentation generators do.
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(0);
  }
  viewer.reset();
  app.shutdown();
  return 0;
}
