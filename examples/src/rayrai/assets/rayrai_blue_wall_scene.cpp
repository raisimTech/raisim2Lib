#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "example_resources.hpp"
#include "raisim/World.hpp"
#include "rayrai/Visuals.hpp"
#include "rayrai/example_common.hpp"

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

bool loadSceneDescription(const std::filesystem::path& path, SceneDescription& scene, std::string& error) {
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
        if (tokens.size() != 2 || tokens[1] != ".") throw std::runtime_error("expected a scene-local asset root");
      } else if (tag == "environment") {
        if (tokens.size() < 12 || environment) throw std::runtime_error("invalid environment record");
        scene.hdrPath = path.parent_path() / tokens[11];
        environment = true;
      } else if (tag == "object") {
        if (tokens.size() < 26 || mesh || tokens[2] != "mesh" || tokens[22] != "visual_only") throw std::runtime_error("expected one visual-only mesh object");
        scene.meshPath = path.parent_path() / tokens[21];
        mesh = true;
      } else if (tag == "camera") {
        if (tokens.size() < 16 || camera) throw std::runtime_error("expected one camera");
        scene.cameraPosition = {std::stof(tokens[2]), std::stof(tokens[3]), std::stof(tokens[4])};
        const glm::quat rotation(std::stof(tokens[5]), std::stof(tokens[6]), std::stof(tokens[7]), std::stof(tokens[8]));
        scene.cameraTarget = scene.cameraPosition + glm::normalize(rotation) * glm::vec3(1.0f, 0.0f, 0.0f);
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
  if (!header || !mesh || !environment || !camera || scene.width < 1 || scene.height < 1 || scene.timeStep <= 0.0 || scene.verticalFov <= 0.0f) {
    error = "Blue Wall .rscene is missing a valid header, mesh, environment, or camera";
    return false;
  }
  return true;
}

void lookAt(raisin::Camera& camera, const SceneDescription& scene) {
  camera.position = scene.cameraPosition;
  camera.target = scene.cameraTarget;
  camera.front = glm::normalize(scene.cameraTarget - scene.cameraPosition);
  camera.up = {0.0f, 0.0f, 1.0f};
  camera.worldUp = camera.up;
  camera.yaw = glm::degrees(std::atan2(camera.front.y, camera.front.x));
  camera.pitch = glm::degrees(std::asin(std::clamp(camera.front.z, -1.0f, 1.0f)));
  camera.zoom = scene.verticalFov;
  camera.nearPlane = camera.zNear = scene.nearPlane;
  camera.farPlane = camera.zFar = scene.farPlane;
  camera.setCameraFixedTarget(false);
  camera.setCameraFixedDistance(false);
}

}  // namespace

int main(int, char* argv[]) {
  const std::filesystem::path scenePath = exampleRscPath(argv[0], "rayrai/blue_wall/blue_wall.rscene");
  SceneDescription description;
  std::string sceneError;
  if (!loadSceneDescription(scenePath, description, sceneError)) {
    std::cerr << sceneError << '\n';
    return 1;
  }
  const std::filesystem::path lightsPath = description.meshPath.string() + ".rayrai_lights.json";
  for (const auto& path : {description.meshPath, lightsPath, description.hdrPath}) {
    if (!std::filesystem::is_regular_file(path)) {
      std::cerr << "Missing Blue Wall asset: " << path << '\n';
      return 1;
    }
  }

  ExampleApp app;
  if (!app.init("RayRai Blue Wall", description.width, description.height)) return 1;

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(description.timeStep);
  auto viewer = std::make_shared<raisin::RayraiWindow>(world, description.width, description.height);
  viewer->setAsyncMeshLoadingEnabled(false);
  auto settings = raisin::RayraiWindow::defaultRenderQualitySettings(raisin::RayraiWindow::RenderQualityPreset::Ultra);
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
  settings.maxPointShadowLights = static_cast<int>(raisin::RayraiWindow::kMaxAdditionalShadowLights);
  settings.additionalShadowResolutionScale = 0.5f;
  settings.pointShadowResolutionScale = 0.5f;
  settings.minAdditionalShadowResolution = 256;
  settings.updateShadowsEveryFrame = false;
  settings.autoSelectImportedShadowLight = true;
  viewer->setRenderQualitySettings(settings);

  raisin::RayraiWindow::SceneImportReport report;
  auto scene = viewer->importVisualScene("blue_wall", description.meshPath.string(), &report, true, 0.0022f);
  if (!scene) {
    std::cerr << "Could not import " << description.meshPath << '\n';
    viewer.reset();
    app.shutdown();
    return 1;
  }
  scene->setTwoSided(false);
  scene->setUseMeshColor(false);
  if (report.shadowLightPromoted || viewer->promoteDominantAdditionalLightToMainShadowCaster(true, true)) {
    viewer->getLight().ambient = settings.mainLightAmbient;
    viewer->getLight().setShadowParams(0.0012f, 0.98f, 1.8f);
    viewer->setShadowOrtho(settings.shadowOrthoHalfSize, 0.05f, 18.0f);
  }

  const std::string hdr = description.hdrPath.string();
  const unsigned int environment = raisin::RayraiWindow::loadHdrEquirectangularCubemap(hdr.c_str(), 64, true);
  const unsigned int irradiance = raisin::RayraiWindow::createHdrIrradianceCubemap(hdr.c_str(), 16, 32);
  const unsigned int prefiltered = raisin::RayraiWindow::createHdrPrefilteredEnvironmentCubemap(hdr.c_str(), 64, 4, 32);
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
  const unsigned int roomMap = viewer->captureReflectionProbeCubemap(probePosition, probeSettings);
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
  lookAt(camera, description);
  std::cout << "Loaded Blue Wall .rscene: " << scenePath << '\n';
  std::cout << "Blue Wall camera: position=(" << camera.position.x << ',' << camera.position.y << ',' << camera.position.z << ") vertical_fov=" << camera.zoom << '\n';

  while (!app.quit) {
    world->integrate();
    app.processEvents();
    app.beginFrame();
    app.renderViewer(*viewer);
    app.endFrame();
  }
  viewer.reset();
  app.shutdown();
  return 0;
}
