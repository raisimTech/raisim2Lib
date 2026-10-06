#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "rayrai/example_common.hpp"
#include "example_resources.hpp"
#include "rayrai_example_compat.hpp"
#include "rayrai/Camera.hpp"
#include "rayrai/CameraFrustum.hpp"
#include "raisim/World.hpp"

namespace
{
ImVec2 fitToAspect(ImVec2 avail, float aspect) {
  if (aspect <= 0.0f)
    return avail;
  ImVec2 size = avail;
  size.y = size.x / aspect;
  if (size.y > avail.y) {
    size.y = avail.y;
    size.x = size.y * aspect;
  }
  return size;
}
} // namespace

static inline glm::vec3 toGlm(const raisim::Vec<3>& v) {
  return glm::vec3(v[0], v[1], v[2]);
}

static inline glm::mat3 toGlm(const raisim::Mat<3, 3>& R) {
  return glm::make_mat3(R.ptr());
}

int main(int argc, char* argv[]) {
  ExampleApp app;
  if (!app.init("rayrai_example_depth", 1280, 720))
    return -1;

  auto world = std::make_shared<raisim::World>();
  world->addGround();

  const std::string sep = raisim::Path::separator();
  const std::string go1Dir = exampleRscPath(argv[0], "go1");
  std::vector<std::string> modules = {"d455"};
  auto go1 = world->addArticulatedSystem(go1Dir + sep + "go1.urdf", modules, go1Dir);
  go1->setGeneralizedCoordinate({0, 0, 0.32, 1.0, 0.0, 0.0, 0.0, 0, 0.67, -1.3, 0, 0.67, -1.3, 0,
    0.67, -1.3, 0, 0.67, -1.3});

  auto depthCam = go1->getSensorSet("d455_front")->getSensor<raisim::DepthCamera>("depth");

  auto sphere = world->addSphere(0.2, 1);
  sphere->setAppearance("0.9,0.3,0.2,1.0");

  auto box = world->addBox(0.4, 0.4, 0.4, 1);
  box->setAppearance("0.2,0.7,0.9,1.0");

  auto viewer = std::make_shared<raisin::RayraiWindow>(world, 1280, 720);
  viewer->setRenderQualitySettings(raisin::RayraiWindow::defaultRenderQualitySettings(
    raisin::RayraiWindow::RenderQualityPreset::Balanced));
  raisim_examples::setRayraiBackgroundColorRgb255(*viewer, {35, 35, 45, 255});
  auto& camera = viewer->getCamera();
  camera.position = {3.2f, 2.6f, 2.2f};
  camera.yaw = -135.0f;
  camera.pitch = -28.0f;

  auto depthCamera = std::make_shared<raisin::Camera>(*depthCam);
  auto depthFrustum = viewer->addCameraFrustum("depth_frustum", glm::vec4(1.0f, 0.6f, 0.2f, 0.4f));
  depthFrustum->setDetectable(false);

  const auto& depthProp = depthCam->getProperties();
  const int depthWidth = std::max(1, depthProp.width);
  const int depthHeight = std::max(1, depthProp.height);
  std::vector<float> rayraiDepth(size_t(depthWidth) * size_t(depthHeight));
  float centerRayraiDepth = std::numeric_limits<float>::quiet_NaN();

  // Sensed points drawn as one instanced batch of small cubes, shaded from yellow at the near clip
  // to blue kPointColorRange beyond it. Every kPointStride-th pixel in each direction becomes a
  // cube. The batch is not detectable, so the depth camera never sees its own points.
  constexpr int kPointStride = 4;
  constexpr float kPointCubeSize = 0.015f;
  constexpr float kPointColorRange = 3.0f;
  auto depthPoints = viewer->addInstancedVisuals("depth_points", raisim::Shape::Box,
    glm::vec3(kPointCubeSize), glm::vec4(1.0f, 0.85f, 0.2f, 1.0f),
    glm::vec4(0.2f, 0.4f, 1.0f, 1.0f));
  depthPoints->setDetectable(false);
  depthPoints->setCastsShadows(false);
  const float clipNear = float(depthProp.clipNear);
  std::vector<raisim::Vec<3>> sensedPoints;
  std::vector<raisin::InstancedVisuals::InstanceSpec> pointInstances;
  pointInstances.reserve(size_t((depthWidth + kPointStride - 1) / kPointStride) *
                         size_t((depthHeight + kPointStride - 1) / kPointStride));

  depthCam->updatePose();
  const glm::vec3 sensorPos = toGlm(depthCam->getPosition());
  const glm::mat3 sensorRot = toGlm(depthCam->getOrientation());
  glm::vec3 forward = glm::normalize(glm::vec3(sensorRot[0]));
  glm::vec3 up = glm::normalize(glm::vec3(sensorRot[2]));
  glm::vec3 right = glm::normalize(glm::cross(forward, up));
  if (glm::dot(right, right) < 1e-6f) {
    right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 0.0f, 1.0f)));
  }
  up = glm::normalize(glm::cross(right, forward));

  const glm::vec3 base = sensorPos + forward * 2.0f;
  const glm::vec3 spherePos = base + right * 0.4f + up * 0.2f;
  const glm::vec3 boxPos = base - right * 0.4f - up * 0.1f;
  sphere->setPosition(spherePos.x, spherePos.y, spherePos.z);
  box->setPosition(boxPos.x, boxPos.y, boxPos.z);

  while (!app.quit) {
    app.processEvents();
    if (app.quit)
      break;

    world->integrate();

    viewer->renderWithExternalCamera(*depthCam, *depthCamera, {});
    viewer->renderDepthPlaneDistance(*depthCam, *depthCamera);
    // Flip so rows run top to bottom, the pixel layout DepthCamera uses for its rays.
    depthCamera->getRawImage(*depthCam,
                             raisin::Camera::SensorStorageMode::CUSTOM_BUFFER,
                             rayraiDepth.data(),
                             rayraiDepth.size(),
                             /*flipVertical=*/true);
    centerRayraiDepth = rayraiDepth[size_t(depthHeight / 2) * size_t(depthWidth) + size_t(depthWidth / 2)];

    // Back-project the depth image into world-frame points and show them as cubes.
    depthCam->depthToPointCloud(rayraiDepth, sensedPoints);
    pointInstances.clear();
    for (int row = 0; row < depthHeight; row += kPointStride) {
      for (int col = 0; col < depthWidth; col += kPointStride) {
        const size_t pixel = size_t(row) * size_t(depthWidth) + size_t(col);
        const float depth = rayraiDepth[pixel];
        // rayrai writes 0 where the ray hit nothing.
        if (!(depth > 0.0f))
          continue;
        raisin::InstancedVisuals::InstanceSpec point;
        point.position = toGlm(sensedPoints[pixel]);
        point.colorWeight = (depth - clipNear) / kPointColorRange;
        pointInstances.push_back(point);
      }
    }
    depthPoints->clearInstances();
    depthPoints->addInstances(pointInstances.data(), pointInstances.size());

    // CPU depth camera path.
    //
    // This is disabled by default so the example remains buildable with older
    // installed RaiSim packages. Uncomment this block after building against a
    // RaiSim library that exposes World::captureDepthCamera. It does not use
    // rayrai or the TCP visualizer. It casts CPU rays against collision geometry
    // and returns depth, object segmentation ids, optional world-frame hit
    // points, a timestamp, and deterministic depth noise.
#if 0
    depthCam->updatePose();
    raisim::World::DepthCameraProperties cpuProp;
    cpuProp.width = depthWidth;
    cpuProp.height = depthHeight;
    cpuProp.clipNear = depthCam->getProperties().clipNear;
    cpuProp.clipFar = depthCam->getProperties().clipFar;
    cpuProp.hFOV = depthCam->getProperties().hFOV;
    cpuProp.captureDepth = true;
    cpuProp.captureSegmentation = true;
    cpuProp.capturePoints = true;
    cpuProp.depthNoiseType = raisim::World::DepthCameraProperties::DepthNoiseType::GAUSSIAN;
    cpuProp.depthNoiseMean = 0.0;
    cpuProp.depthNoiseStd = 0.002;
    cpuProp.depthNoiseSeed = 7;

    raisim::World::DepthCameraFrame cpuFrame;
    world->captureDepthCamera(depthCam->getPosition(),
                              depthCam->getOrientation(),
                              cpuProp,
                              cpuFrame,
                              go1->getIndexInWorld(),
                              depthCam->getFrameId());

    const int cpuCenter = (cpuProp.height / 2) * cpuProp.width + (cpuProp.width / 2);
    const auto& hitPoint = cpuFrame.points[cpuCenter];
    RSINFO("cpu depth camera t=" << cpuFrame.timeStamp
           << " center_depth=" << cpuFrame.depth[cpuCenter]
           << " center_object_id=" << cpuFrame.segmentation[cpuCenter]
           << " center_hit_point=[" << hitPoint[0] << ", " << hitPoint[1] << ", " << hitPoint[2] << "]")
#endif

    if (depthFrustum) {
      depthFrustum->updateFromDepthCamera(*depthCam);
    }

    app.beginFrame();
    app.renderViewer(*viewer);

    const float aspect = float(depthWidth) / float(depthHeight);

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 270), ImGuiCond_FirstUseEver);
    ImGui::Begin("Depth Sensor", nullptr, ImGuiWindowFlags_NoCollapse);
    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.y = std::max(1.0f, avail.y - 2.0f * ImGui::GetTextLineHeightWithSpacing());
    ImVec2 size = fitToAspect(avail, aspect);
    ImTextureID tex = (ImTextureID)(intptr_t)depthCamera->getLinearDepthTexture();
    ImGui::Image(tex, size, ImVec2(0, 1), ImVec2(1, 0));
    ImGui::Text("rayrai center depth: %.4f m", centerRayraiDepth);
    ImGui::Text("sensed points: %zu cubes", depthPoints->count());
    ImGui::End();

    app.endFrame();
  }

  viewer.reset();
  app.shutdown();
  return 0;
}
