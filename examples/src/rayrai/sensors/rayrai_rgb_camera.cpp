#include <algorithm>
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
  if (!app.init("rayrai_example_rgb", 1280, 720)) return -1;

  auto world = std::make_shared<raisim::World>();
  world->addGround();

  const std::string sep = raisim::Path::separator();
  const std::string go1Dir = exampleRscPath(argv[0], "go1");
  std::vector<std::string> modules = {"d455"};
  raisim::ArticulatedSystemOption options;
  options.convexifyCollisionMeshes = true;
  auto go1 = world->addArticulatedSystem(go1Dir + sep + "go1.urdf", modules, go1Dir, {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);
  go1->setGeneralizedCoordinate({0, 0, 0.32, 1.0, 0.0, 0.0, 0.0, 0, 0.67, -1.3, 0, 0.67, -1.3, 0, 0.67, -1.3, 0, 0.67, -1.3});

  auto rgbCam = go1->getSensorSet("d455_front")->getSensor<raisim::RGBCamera>("color");

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

  const auto rgbProp = rgbCam->getProperties();
  auto rgbCamera = std::make_shared<raisin::Camera>(rgbProp);
  auto rgbFrustum = viewer->addCameraFrustum("rgb_frustum", glm::vec4(0.2f, 0.7f, 1.0f, 0.4f));
  rgbFrustum->setDetectable(false);
  std::vector<char> rgbImage(size_t(rgbProp.width) * size_t(rgbProp.height) * 4);

  rgbCam->updatePose();
  const glm::vec3 sensorPos = raisim_examples::toGlm(rgbCam->getPosition());
  const glm::mat3 sensorRot = raisim_examples::toGlm(rgbCam->getOrientation());
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
    rgbCam->updatePose();
    const glm::vec3 capturedPosition = raisim_examples::toGlm(rgbCam->getPosition());
    const glm::mat3 capturedRotation = raisim_examples::toGlm(rgbCam->getOrientation());

    // Only copied data is used below. A separate render thread would lock the window here.
    viewer->applyPhysicsSnapshot(snapshot);
    rgbCamera->position = capturedPosition;
    rgbCamera->front = capturedRotation[0];  // Sensor +X is forward, +Z is up.
    rgbCamera->up = capturedRotation[2];
    rgbCamera->right = glm::normalize(glm::cross(rgbCamera->front, rgbCamera->up));

    viewer->renderWithExternalCamera(*rgbCamera, {});
    rgbCamera->readSceneColorBgra(rgbImage.data(), rgbImage.size());
    if (rgbFrustum) {
      rgbFrustum->updateFromCamera(*rgbCamera);
    }

    app.beginFrame();
    app.renderViewer(*viewer);

    const auto& prop = rgbProp;
    const int w = std::max(1, prop.width);
    const int h = std::max(1, prop.height);
    const float aspect = float(w) / float(h);

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 220), ImGuiCond_FirstUseEver);
    ImGui::Begin("RGB Sensor", nullptr, ImGuiWindowFlags_NoCollapse);
    raisim_examples::drawFittedImage(rgbCamera->getFinalTexture(), aspect, 1);
    const auto center = (size_t(h / 2) * size_t(w) + size_t(w / 2)) * 4;
    ImGui::Text("Center RGB: %u %u %u", unsigned(static_cast<unsigned char>(rgbImage[center + 2])), unsigned(static_cast<unsigned char>(rgbImage[center + 1])),
                unsigned(static_cast<unsigned char>(rgbImage[center])));
    ImGui::End();

    app.endFrame();
  }

  rgbCamera.reset();
  rgbFrustum.reset();
  viewer.reset();
  app.shutdown();
  return 0;
}
