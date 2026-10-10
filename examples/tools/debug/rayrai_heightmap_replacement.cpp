// Manual regression tool for stale geometry in depth images after scene replacement.

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <glm/glm.hpp>

#include "rayrai/example_common.hpp"
#include "rayrai/Camera.hpp"
#include "rayrai/CameraFrustum.hpp"
#include "rayrai_example_compat.hpp"
#include "example_resources.hpp"
#include "raisim/World.hpp"

namespace
{
constexpr size_t kTerrainSamples = 129;
constexpr double kTerrainSize = 12.0;

// Projected depth pixels are drawn as small cubes, one for every kPointStride-th pixel in each
// direction, shaded from the near clip to kPointColorRange beyond it.
constexpr int kPointStride = 6;
constexpr float kPointCubeSize = 0.012f;
constexpr float kPointColorRange = 3.0f;

ImVec2 fitToAspect(ImVec2 available, float aspect) {
  if (aspect <= 0.0f)
    return available;

  ImVec2 size = available;
  size.y = size.x / aspect;
  if (size.y > available.y) {
    size.y = available.y;
    size.x = size.y * aspect;
  }
  return size;
}

// Projects a depth image (rows top to bottom) into world-frame points and replaces the cubes in
// `cubes` with them. Pixels where the camera saw nothing (depth 0) are skipped.
void showProjectedPixels(raisim::DepthCamera& sensor, const std::vector<float>& depth,
                         std::vector<raisim::Vec<3>>& points,
                         std::vector<raisin::InstancedVisuals::InstanceSpec>& instances,
                         raisin::InstancedVisuals& cubes) {
  const auto& properties = sensor.getProperties();
  sensor.depthToPointCloud(depth, points);
  instances.clear();
  for (int row = 0; row < properties.height; row += kPointStride) {
    for (int col = 0; col < properties.width; col += kPointStride) {
      const size_t pixel = size_t(row) * size_t(properties.width) + size_t(col);
      if (!(depth[pixel] > 0.0f))
        continue;
      raisin::InstancedVisuals::InstanceSpec cube;
      cube.position = glm::vec3(points[pixel][0], points[pixel][1], points[pixel][2]);
      cube.colorWeight = (depth[pixel] - float(properties.clipNear)) / kPointColorRange;
      instances.push_back(cube);
    }
  }
  cubes.clearInstances();
  cubes.addInstances(instances.data(), instances.size());
}

double smoothStep(double edge0, double edge1, double value) {
  const double t = std::clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

std::vector<double> makeTerrainHeights(int generation) {
  std::vector<double> heights(kTerrainSamples * kTerrainSamples, 0.0);
  const double spacing = kTerrainSize / double(kTerrainSamples - 1);
  const double phase = 1.37 * double(generation);

  // The front and rear cameras are pitched down and meet the flat terrain near
  // x=+/-1.4. Keep a prominent mound just before each of those points so the
  // terrain cannot hide it. Moving the mounds left/right makes stale geometry
  // from an earlier generation easy to distinguish in both depth images.
  const double frontMoundX = 1.35;
  const double rearMoundX = -1.35;
  constexpr double moundYByGeneration[] = {-0.32, 0.32, 0.0};
  const double frontMoundY = moundYByGeneration[generation % 3];
  const double rearMoundY = -frontMoundY;
  const double moundHeight = generation % 2 == 0 ? 0.78 : 0.58;
  const bool xBands = generation % 2 == 0;

  for (size_t yIndex = 0; yIndex < kTerrainSamples; ++yIndex) {
    const double y = -0.5 * kTerrainSize + spacing * double(yIndex);
    for (size_t xIndex = 0; xIndex < kTerrainSamples; ++xIndex) {
      const double x = -0.5 * kTerrainSize + spacing * double(xIndex);
      const double radius = std::hypot(x, y);

      // Change both the phase and direction of the large terrain features so
      // consecutive maps look substantially different in the main view.
      const double rolling =
        0.16 * std::sin(0.72 * x + phase) * std::cos(0.83 * y - 0.55 * phase);
      const double bands = 0.07 * std::sin(2.4 * (xBands ? x : y) + phase);
      const double frontMoundDx = (x - frontMoundX) / 0.30;
      const double frontMoundDy = (y - frontMoundY) / 0.30;
      const double rearMoundDx = (x - rearMoundX) / 0.30;
      const double rearMoundDy = (y - rearMoundY) / 0.30;
      const double frontMound = moundHeight *
        std::exp(-(frontMoundDx * frontMoundDx + frontMoundDy * frontMoundDy));
      const double rearMound = moundHeight *
        std::exp(-(rearMoundDx * rearMoundDx + rearMoundDy * rearMoundDy));

      // Keep a generous, exactly-flat patch below and around the robot. The
      // blend avoids a sharp collision edge at the boundary of that patch.
      const double flatBlend = smoothStep(1.15, 1.55, radius);
      heights[yIndex * kTerrainSamples + xIndex] =
        flatBlend * (rolling + bands + frontMound + rearMound);
    }
  }
  return heights;
}
} // namespace

int main(int argc, char* argv[]) {
  ExampleApp app;
  if (!app.init("rayrai heightmap replacement", 1280, 720))
    return -1;

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(0.002);

  const std::string anymalUrdf =
    exampleRscPath(argv[0], "anymal_c/urdf/anymal_sensored.urdf");
  auto* anymal = world->addArticulatedSystem(anymalUrdf);

  Eigen::VectorXd jointTarget(anymal->getGeneralizedCoordinateDim());
  Eigen::VectorXd velocityTarget(anymal->getDOF());
  Eigen::VectorXd pGain(anymal->getDOF());
  Eigen::VectorXd dGain(anymal->getDOF());
  jointTarget << 0.0, 0.0, 0.54, 1.0, 0.0, 0.0, 0.0,
    0.03, 0.4, -0.8, -0.03, 0.4, -0.8,
    0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
  velocityTarget.setZero();
  pGain.setZero();
  dGain.setZero();
  pGain.tail(12).setConstant(200.0);
  dGain.tail(12).setConstant(10.0);

  anymal->setState(jointTarget, velocityTarget);
  anymal->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  anymal->setPdGains(pGain, dGain);
  anymal->setPdTarget(jointTarget, velocityTarget);
  anymal->setGeneralizedForce(Eigen::VectorXd::Zero(anymal->getDOF()));
  anymal->setName("anymal_depth_camera");

  auto* frontDepthCam = anymal->getSensorSet("depth_camera_front_camera_parent")
    ->getSensor<raisim::DepthCamera>("depth");
  frontDepthCam->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);
  auto* rearDepthCam = anymal->getSensorSet("depth_camera_rear_camera_parent")
    ->getSensor<raisim::DepthCamera>("depth");
  rearDepthCam->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);

  int terrainGeneration = 0;
  auto addTerrain = [&](int generation) {
    auto heights = makeTerrainHeights(generation);
    auto* terrain = world->addHeightMap(
      kTerrainSamples, kTerrainSamples, kTerrainSize, kTerrainSize,
      0.0, 0.0, heights);
    terrain->setName("replaceable_heightmap_" + std::to_string(generation));
    terrain->setAppearance(generation % 2 == 0
      ? "0.18,0.52,0.22,1.0"
      : "0.62,0.32,0.10,1.0");
    return terrain;
  };
  raisim::HeightMap* terrain = addTerrain(terrainGeneration);

  auto addCameraPrimitives = [&](int generation, double direction,
                                 const std::string& cameraName) {
    std::vector<raisim::SingleBodyObject*> objects;
    const unsigned int sideSeed = direction > 0.0 ? 0x13579u : 0x24680u;
    std::mt19937 randomGenerator(
      0xC0FFEEu + 7919u * unsigned(generation) + sideSeed);
    std::uniform_int_distribution<int> objectCountDistribution(2, 9);
    std::uniform_int_distribution<int> shapeDistribution(0, 2);
    std::uniform_int_distribution<int> colorDistribution(0, 4);
    std::uniform_real_distribution<double> distanceDistribution(0.95, 1.28);
    std::uniform_real_distribution<double> lateralDistribution(-0.30, 0.30);
    std::uniform_real_distribution<double> sizeDistribution(0.12, 0.24);

    constexpr const char* colors[] = {
      "0.90,0.22,0.18,1.0",
      "0.15,0.58,0.95,1.0",
      "0.96,0.72,0.12,1.0",
      "0.62,0.28,0.88,1.0",
      "0.12,0.78,0.55,1.0"
    };

    const int objectCount = objectCountDistribution(randomGenerator);
    objects.reserve(size_t(objectCount));
    for (int index = 0; index < objectCount; ++index) {
      const double x = direction * distanceDistribution(randomGenerator);
      const double y = lateralDistribution(randomGenerator);
      const double size = sizeDistribution(randomGenerator);
      const int shape = shapeDistribution(randomGenerator);

      raisim::SingleBodyObject* object = nullptr;
      double centerHeight = 0.0;
      if (shape == 0) {
        const double zSize = 1.6 * size;
        object = world->addBox(size, 0.85 * size, zSize, 1.0);
        centerHeight = 0.5 * zSize;
      } else if (shape == 1) {
        object = world->addSphere(0.55 * size, 1.0);
        centerHeight = 0.55 * size;
      } else {
        const double cylinderHeight = 1.4 * size;
        object = world->addCylinder(0.45 * size, cylinderHeight, 1.0);
        centerHeight = 0.5 * cylinderHeight;
      }

      object->setName(cameraName + "_primitive_g" +
                      std::to_string(generation) + "_" + std::to_string(index));
      object->setAppearance(colors[colorDistribution(randomGenerator)]);
      object->setBodyType(raisim::BodyType::STATIC);
      object->setPosition(
        x, y, terrain->getHeight(x, y) + centerHeight + 0.01);
      objects.push_back(object);
    }
    return objects;
  };

  auto frontPrimitives = addCameraPrimitives(terrainGeneration, 1.0, "front");
  auto rearPrimitives = addCameraPrimitives(terrainGeneration, -1.0, "rear");

  auto viewer = std::make_shared<raisin::RayraiWindow>(world, 1280, 720);
  viewer->setRenderQualitySettings(raisin::RayraiWindow::defaultRenderQualitySettings(
    raisin::RayraiWindow::RenderQualityPreset::Balanced));
  raisim_examples::setRayraiBackgroundColorRgb255(*viewer, {32, 35, 42, 255});
  raisim_examples::addRayraiBasicSceneLights(*viewer);

  auto& viewerCamera = viewer->getCamera();
  viewerCamera.target = {0.0f, 0.0f, 0.2f};
  viewerCamera.position = {3.4f, -4.2f, 2.6f};
  viewerCamera.yaw = 128.0f;
  viewerCamera.pitch = -25.0f;
  viewerCamera.setCameraFixedTarget(true);
  viewerCamera.setCameraFixedDistance(true);

  auto frontDepthCamera = std::make_shared<raisin::Camera>(*frontDepthCam);
  auto rearDepthCamera = std::make_shared<raisin::Camera>(*rearDepthCam);
  auto frontDepthFrustum = viewer->addCameraFrustum(
    "front_depth_frustum", glm::vec4(1.0f, 0.62f, 0.18f, 0.45f));
  auto rearDepthFrustum = viewer->addCameraFrustum(
    "rear_depth_frustum", glm::vec4(0.20f, 0.72f, 1.0f, 0.45f));
  frontDepthFrustum->setDetectable(false);
  rearDepthFrustum->setDetectable(false);

  // Projected pixels of each camera, bright near the camera and dark far away, in the color of
  // its frustum. They are not detectable, so the cameras never see each other's cubes.
  auto addProjectedPixelCubes = [&](const std::string& name, const glm::vec3& color) {
    auto cubes = viewer->addInstancedVisuals(name, raisim::Shape::Box,
      glm::vec3(kPointCubeSize), glm::vec4(color, 1.0f), glm::vec4(0.35f * color, 1.0f));
    cubes->setDetectable(false);
    cubes->setCastsShadows(false);
    return cubes;
  };
  auto frontPixelCubes = addProjectedPixelCubes("front_depth_pixels", {1.0f, 0.62f, 0.18f});
  auto rearPixelCubes = addProjectedPixelCubes("rear_depth_pixels", {0.20f, 0.72f, 1.0f});
  std::vector<raisim::Vec<3>> projectedPoints;
  std::vector<raisin::InstancedVisuals::InstanceSpec> pixelCubeInstances;

  const auto& frontDepthProperties = frontDepthCam->getProperties();
  const int frontDepthWidth = std::max(1, frontDepthProperties.width);
  const int frontDepthHeight = std::max(1, frontDepthProperties.height);
  std::vector<float> frontDepthValues(
    size_t(frontDepthWidth) * size_t(frontDepthHeight));
  float frontCenterDepth = std::numeric_limits<float>::quiet_NaN();

  const auto& rearDepthProperties = rearDepthCam->getProperties();
  const int rearDepthWidth = std::max(1, rearDepthProperties.width);
  const int rearDepthHeight = std::max(1, rearDepthProperties.height);
  std::vector<float> rearDepthValues(
    size_t(rearDepthWidth) * size_t(rearDepthHeight));
  float rearCenterDepth = std::numeric_limits<float>::quiet_NaN();
  bool spaceWasDown = false;

  while (!app.quit) {
    app.processEvents();
    if (app.quit)
      break;

    const bool spaceIsDown = SDL_GetKeyboardState(nullptr)[SDL_SCANCODE_SPACE] != 0;
    if (spaceIsDown && !spaceWasDown) {
      // Allocate the replacement while the old map is still alive so it must
      // have a distinct address. Then remove the old map before rendering.
      // This avoids allocator pointer reuse masking the replacement itself.
      auto* oldTerrain = terrain;
      auto oldFrontPrimitives = frontPrimitives;
      auto oldRearPrimitives = rearPrimitives;
      terrain = addTerrain(++terrainGeneration);
      frontPrimitives = addCameraPrimitives(terrainGeneration, 1.0, "front");
      rearPrimitives = addCameraPrimitives(terrainGeneration, -1.0, "rear");
      for (auto* object : oldFrontPrimitives)
        world->removeObject(object);
      for (auto* object : oldRearPrimitives)
        world->removeObject(object);
      world->removeObject(oldTerrain);
    }
    spaceWasDown = spaceIsDown;

    world->integrate();

    viewer->renderWithExternalCamera(*frontDepthCam, *frontDepthCamera, {});
    viewer->renderDepthPlaneDistance(*frontDepthCam, *frontDepthCamera);
    // Flip so rows run top to bottom, the pixel layout DepthCamera uses for its rays.
    frontDepthCamera->getRawImage(
      *frontDepthCam, raisin::Camera::SensorStorageMode::CUSTOM_BUFFER,
      frontDepthValues.data(), frontDepthValues.size(), /*flipVertical=*/true);
    frontCenterDepth = frontDepthValues[
      size_t(frontDepthHeight / 2) * size_t(frontDepthWidth) +
      size_t(frontDepthWidth / 2)];
    showProjectedPixels(*frontDepthCam, frontDepthValues, projectedPoints,
                        pixelCubeInstances, *frontPixelCubes);

    viewer->renderWithExternalCamera(*rearDepthCam, *rearDepthCamera, {});
    viewer->renderDepthPlaneDistance(*rearDepthCam, *rearDepthCamera);
    rearDepthCamera->getRawImage(
      *rearDepthCam, raisin::Camera::SensorStorageMode::CUSTOM_BUFFER,
      rearDepthValues.data(), rearDepthValues.size(), /*flipVertical=*/true);
    rearCenterDepth = rearDepthValues[
      size_t(rearDepthHeight / 2) * size_t(rearDepthWidth) +
      size_t(rearDepthWidth / 2)];
    showProjectedPixels(*rearDepthCam, rearDepthValues, projectedPoints,
                        pixelCubeInstances, *rearPixelCubes);

    if (frontDepthFrustum)
      frontDepthFrustum->updateFromDepthCamera(*frontDepthCam);
    if (rearDepthFrustum)
      rearDepthFrustum->updateFromDepthCamera(*rearDepthCam);

    app.beginFrame();
    app.renderViewer(*viewer);

    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::SetNextWindowSize(ImVec2(440, 700));
    ImGui::Begin("Heightmap replacement depth", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoCollapse);
    ImGui::Text("Tap SPACE to delete and recreate the heightmap.");
    ImGui::Text("Current generation: %d", terrainGeneration);
    ImGui::Separator();

    const ImVec2 panelAvailable = ImGui::GetContentRegionAvail();
    const float panelSpacing = ImGui::GetStyle().ItemSpacing.y;
    const float depthPanelHeight =
      std::max(1.0f, 0.5f * (panelAvailable.y - panelSpacing));

    ImGui::BeginChild("FrontDepthPanel",
      ImVec2(panelAvailable.x, depthPanelHeight), true,
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::Text("Front depth | primitives: %zu | pixel cubes: %zu",
                frontPrimitives.size(), frontPixelCubes->count());
    ImGui::Text("Center depth: %.4f m", frontCenterDepth);
    ImVec2 frontAvailable = ImGui::GetContentRegionAvail();
    const float frontAspect = float(frontDepthWidth) / float(frontDepthHeight);
    ImVec2 frontImageSize = fitToAspect(frontAvailable, frontAspect);
    ImTextureID frontTexture =
      (ImTextureID)(intptr_t)frontDepthCamera->getLinearDepthTexture();
    ImGui::Image(frontTexture, frontImageSize, ImVec2(0, 1), ImVec2(1, 0));
    ImGui::EndChild();

    ImGui::Dummy(ImVec2(0.0f, panelSpacing));
    ImGui::BeginChild("RearDepthPanel", ImVec2(panelAvailable.x, 0.0f), true,
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::Text("Rear depth | primitives: %zu | pixel cubes: %zu",
                rearPrimitives.size(), rearPixelCubes->count());
    ImGui::Text("Center depth: %.4f m", rearCenterDepth);
    ImVec2 rearAvailable = ImGui::GetContentRegionAvail();
    const float rearAspect = float(rearDepthWidth) / float(rearDepthHeight);
    ImVec2 rearImageSize = fitToAspect(rearAvailable, rearAspect);
    ImTextureID rearTexture =
      (ImTextureID)(intptr_t)rearDepthCamera->getLinearDepthTexture();
    ImGui::Image(rearTexture, rearImageSize, ImVec2(0, 1), ImVec2(1, 0));
    ImGui::EndChild();
    ImGui::End();

    app.endFrame();
  }

  viewer.reset();
  app.shutdown();
  return 0;
}
