#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "example_resources.hpp"
#include "raisim/World.hpp"
#include "rayrai/Camera.hpp"
#include "rayrai/CameraFrustum.hpp"
#include "rayrai/PhysicsSnapshot.hpp"
#include "rayrai/example_common.hpp"
#include "rayrai_example_compat.hpp"
#include "rayrai_example_utils.hpp"

int main(int argc, char* argv[]) {
  ExampleApp app;
  if (!app.init("rayrai_example_depth", 1280, 720)) return -1;

  auto world = std::make_shared<raisim::World>();
  world->addGround();

  const std::string sep = raisim::Path::separator();
  const std::string go1Dir = exampleRscPath(argv[0], "go1");
  std::vector<std::string> modules = {"d455"};
  raisim::ArticulatedSystemOption options;
  options.convexifyCollisionMeshes = true;
  auto go1 = world->addArticulatedSystem(go1Dir + sep + "go1.urdf", modules, go1Dir, {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);
  go1->setGeneralizedCoordinate({0, 0, 0.32, 1.0, 0.0, 0.0, 0.0, 0, 0.67, -1.3, 0, 0.67, -1.3, 0, 0.67, -1.3, 0, 0.67, -1.3});

  auto depthCam = go1->getSensorSet("d455_front")->getSensor<raisim::DepthCamera>("depth");

  auto sphere = world->addSphere(0.2, 1);
  sphere->setAppearance("0.9,0.3,0.2,1.0");

  auto box = world->addBox(0.4, 0.4, 0.4, 1);
  box->setAppearance("0.2,0.7,0.9,1.0");

  // The renderer owns an empty world; physics reaches it only through copied snapshots.
  raisin::PhysicsSnapshotCapture snapshotCapture;
  auto viewer = std::make_shared<raisin::RayraiWindow>(std::make_shared<raisim::World>(), 1280, 720);
  viewer->setRenderQualitySettings(raisin::RayraiWindow::defaultRenderQualitySettings(raisin::RayraiWindow::RenderQualityPreset::Balanced));
  raisim_examples::setRayraiBackgroundColorRgb255(*viewer, {35, 35, 45, 255});
  auto& camera = viewer->getCamera();
  camera.position = {3.2f, 2.6f, 2.2f};
  camera.yaw = -135.0f;
  camera.pitch = -28.0f;

  const auto depthProp = depthCam->getProperties();
  auto depthCamera = std::make_shared<raisin::Camera>(depthProp);
  auto depthFrustum = viewer->addCameraFrustum("depth_frustum", glm::vec4(1.0f, 0.6f, 0.2f, 0.4f));
  depthFrustum->setDetectable(false);

  // Copy calibration rays once, so depth back-projection also avoids the live sensor.
  const auto sensorRays = depthCam->getPrecomputedRayDir();
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
  auto depthPoints = viewer->addInstancedVisuals("depth_points", raisim::Shape::Box, glm::vec3(kPointCubeSize), glm::vec4(1.0f, 0.85f, 0.2f, 1.0f), glm::vec4(0.2f, 0.4f, 1.0f, 1.0f));
  depthPoints->setDetectable(false);
  depthPoints->setCastsShadows(false);
  const float clipNear = float(depthProp.clipNear);
  std::vector<raisin::InstancedVisuals::InstanceSpec> pointInstances;
  pointInstances.reserve(size_t((depthWidth + kPointStride - 1) / kPointStride) * size_t((depthHeight + kPointStride - 1) / kPointStride));

  depthCam->updatePose();
  const glm::vec3 sensorPos = raisim_examples::toGlm(depthCam->getPosition());
  const glm::mat3 sensorRot = raisim_examples::toGlm(depthCam->getOrientation());
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
    if (app.quit) break;

    world->integrate();
    // CPU-only copy immediately after physics; apply on the thread owning the GL context.
    const auto snapshot = snapshotCapture.capture(*world);
    depthCam->updatePose();
    const glm::vec3 capturedPosition = raisim_examples::toGlm(depthCam->getPosition());
    const glm::mat3 capturedRotation = raisim_examples::toGlm(depthCam->getOrientation());

    // Only copied data is used below. A separate render thread would lock the window here.
    viewer->applyPhysicsSnapshot(snapshot);
    depthCamera->position = capturedPosition;
    depthCamera->front = capturedRotation[0];  // Sensor +X is forward, +Z is up.
    depthCamera->up = capturedRotation[2];
    depthCamera->right = glm::normalize(glm::cross(depthCamera->front, depthCamera->up));

    viewer->renderWithExternalCamera(*depthCamera, {});
    viewer->renderDepthPlaneDistance(*depthCamera);
    // Flip so rows run top to bottom, the pixel layout DepthCamera uses for its rays.
    depthCamera->readLinearDepth(rayraiDepth.data(), rayraiDepth.size(), /*flipVertical=*/true);
    centerRayraiDepth = rayraiDepth[size_t(depthHeight / 2) * size_t(depthWidth) + size_t(depthWidth / 2)];

    // Back-project the depth image into world-frame points and show them as cubes.
    pointInstances.clear();
    for (int row = 0; row < depthHeight; row += kPointStride) {
      for (int col = 0; col < depthWidth; col += kPointStride) {
        const size_t pixel = size_t(row) * size_t(depthWidth) + size_t(col);
        const float depth = rayraiDepth[pixel];
        // rayrai writes 0 where the ray hit nothing.
        if (!std::isfinite(depth) || !(depth > 0.0f)) continue;
        const auto ray = raisim_examples::toGlm(sensorRays[pixel]);
        if (!(ray.x > 0.0f)) continue;
        raisin::InstancedVisuals::InstanceSpec point;
        // Plane depth is measured along sensor +X, including for fisheye rays.
        point.position = capturedPosition + capturedRotation * (ray * (depth / ray.x));
        point.colorWeight = (depth - clipNear) / kPointColorRange;
        pointInstances.push_back(point);
      }
    }
    depthPoints->clearInstances();
    depthPoints->addInstances(pointInstances.data(), pointInstances.size());

    // Optional CPU depth camera path: run on the physics thread under the World lock.
    // Keep this independent of the snapshot rendering above.
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
      depthFrustum->updateFromCamera(*depthCamera);
    }

    app.beginFrame();
    app.renderViewer(*viewer);

    const float aspect = float(depthWidth) / float(depthHeight);

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 270), ImGuiCond_FirstUseEver);
    ImGui::Begin("Depth Sensor", nullptr, ImGuiWindowFlags_NoCollapse);
    raisim_examples::drawFittedImage(depthCamera->getLinearDepthTexture(), aspect, 2);
    ImGui::Text("rayrai center depth: %.4f m", centerRayraiDepth);
    ImGui::Text("sensed points: %zu cubes", depthPoints->count());
    ImGui::End();

    app.endFrame();
  }

  depthCamera.reset();
  depthFrustum.reset();
  depthPoints.reset();
  viewer.reset();
  app.shutdown();
  return 0;
}
