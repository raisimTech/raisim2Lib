// A photoreal warehouse loaded from rayrai_warehouse.rscene, with an ANYmal C
// quadruped standing in an aisle between pallet racks. RaiSim builds the
// building, rack and load colliders, the props and a few loose cartons and
// crates from the file; the .rscene reader creates no articulated systems, so
// the robot is added here.
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <glbinding/gl/gl.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "rayrai/example_common.hpp"
#include "rayrai/RsceneVisuals.hpp"
#include "example_resources.hpp"
#include "rayrai_loading_progress.hpp"

namespace {

// Matches ROBOT_XY in examples/tools/generate_warehouse_rscene.py, which keeps
// this spot of the middle aisle free.
constexpr double kRobotX = 4.4, kRobotY = 0.3, kRobotYaw = -0.6;

raisim::ArticulatedSystem* addStandingAnymal(raisim::World& world, const std::string& urdf) {
  auto* anymal = world.addArticulatedSystem(urdf);
  anymal->setName("anymal");
  Eigen::VectorXd gc(anymal->getGeneralizedCoordinateDim()), gv(anymal->getDOF());
  gc << kRobotX, kRobotY, 0.56, std::cos(kRobotYaw / 2), 0, 0, std::sin(kRobotYaw / 2),
    0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
  gv.setZero();
  Eigen::VectorXd pGain = Eigen::VectorXd::Zero(anymal->getDOF());
  Eigen::VectorXd dGain = Eigen::VectorXd::Zero(anymal->getDOF());
  pGain.tail(12).setConstant(200.0);
  dGain.tail(12).setConstant(10.0);
  anymal->setState(gc, gv);
  anymal->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  anymal->setPdGains(pGain, dGain);
  anymal->setPdTarget(gc, gv);
  anymal->setGeneralizedForce(Eigen::VectorXd::Zero(anymal->getDOF()));
  return anymal;
}

bool saveScreenshot(raisin::Camera& camera, const std::filesystem::path& path) {
  const int width = camera.rtWidth(), height = camera.rtHeight();
  const size_t stride = static_cast<size_t>(width) * 4;
  std::vector<unsigned char> rgba(stride * height), flipped(stride * height);
  gl::glBindTexture(gl::GL_TEXTURE_2D, camera.getFinalTexture());
  gl::glGetTexImage(gl::GL_TEXTURE_2D, 0, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
  gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
  for (int y = 0; y < height; ++y)
    std::memcpy(&flipped[y * stride], &rgba[(height - 1 - y) * stride], stride);
  return stbi_write_png(path.string().c_str(), width, height, 4, flipped.data(),
                        static_cast<int>(stride)) != 0;
}

} // namespace

int main(int argc, char* argv[]) {
  std::filesystem::path scene = exampleRscPath(argv[0], "warehouse/rayrai_warehouse.rscene"), screenshot;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--scene" && i + 1 < argc) scene = argv[++i];
    else if (option == "--screenshot" && i + 1 < argc) screenshot = argv[++i];
    else {
      std::cerr << "Usage: rayrai_warehouse [--scene FILE.rscene] [--screenshot PNG]\n";
      return 2;
    }
  }
  auto world = std::make_shared<raisim::World>(scene.string());
  addStandingAnymal(*world, exampleRscPath(argv[0], "anymal_c/urdf/anymal.urdf"));

  ExampleApp app;
  if (!app.init("RaiSim warehouse", 1280, 800, screenshot.empty())) return 1;
  // A capture need not wait for the display, which also makes its frame time a measurement.
  if (!screenshot.empty()) SDL_GL_SetSwapInterval(0);
  ImGui::GetIO().IniFilename = nullptr;
  int status = 0;
  {
    raisin::RayraiWindow viewer(world, 1280, 800);
    viewer.setAsyncMeshLoadingEnabled(true);
    raisin::applyRscene(*world->getRscene(), viewer);
    RayraiLoadingProgress loading("Loading warehouse assets");
    loading.update(viewer.pendingAsyncMeshLoadCount());
    // A screenshot waits for the assets, then lets the robot and the loose props settle.
    constexpr int kSettleFrames = 90;
    int settleFrames = kSettleFrames;
    auto settleStart = std::chrono::steady_clock::now();
    while (!app.quit) {
      app.processEvents();
      if (app.quit) break;
      // Physics starts once the meshes are loaded: eight 2 ms steps per frame.
      if (!loading.visible())
        for (int step = 0; step < 8; ++step) world->integrate();
      app.beginFrame();
      app.renderViewer(viewer);
      loading.update(viewer.pendingAsyncMeshLoadCount());
      if (screenshot.empty()) loading.draw();
      app.endFrame();
      if (!screenshot.empty() && !loading.visible()) {
        if (settleFrames == kSettleFrames) settleStart = std::chrono::steady_clock::now();
        if (--settleFrames > 0) continue;
        const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - settleStart).count();
        std::cout << "Warehouse: " << kSettleFrames - 1 << " frames after loading, "
                  << ms / (kSettleFrames - 1) << " ms per frame\n";
        status = saveScreenshot(viewer.getCamera(), screenshot) ? 0 : 1;
        break;
      }
    }
  } // Release graphics resources before closing the OpenGL context.
  app.shutdown();
  return status;
}
