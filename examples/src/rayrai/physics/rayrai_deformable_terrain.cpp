// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#include <cstdio>
#include <memory>
#include <string>

#include "example_resources.hpp"
#include "rayrai_loading_progress.hpp"
#include "deformable_terrain_scene.hpp"
#include "../worlds/offroad_buggy_visuals.hpp"
#include "raisim/World.hpp"
#include "rayrai/Camera.hpp"
#include "rayrai/example_common.hpp"

namespace {
constexpr int kStepsPerFrame = 16;
}  // namespace

int main(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--activation-key=", 0) != 0) {
      std::fprintf(stderr, "Usage: %s [--activation-key=FILE]\n", argv[0]);
      return 1;
    }
    raisim::World::setActivationKey(arg.substr(17));
  }

  deformable_terrain::Scene scene(exampleRscPath(argv[0], "offroad_buggy/buggy.urdf"));
  auto* car = scene.car;

  ExampleApp app;
  if (!app.init("Wheeled car on deformable mesh soil", 1280, 720)) return 1;
  ImGui::GetIO().IniFilename = nullptr;
  auto viewer = std::make_shared<raisin::RayraiWindow>(scene.world, 1280, 720);
  viewer->setAsyncMeshLoadingEnabled(true);
  deformable_terrain::configureLighting(*viewer);
  auto suspension = std::make_unique<offroad::SuspensionVisuals>(*car);
  auto& camera = viewer->getCamera();
  const auto position = car->getBasePosition();
  camera.position = {float(position[0]) + 4.5f, -5.5f, 2.8f};
  camera.yaw = 140.0f;
  camera.pitch = -18.0f;
  camera.setTargetObject(car);
  camera.setCameraFixedDistance(true);
  RayraiLoadingProgress loading("Loading buggy assets");
  loading.update(viewer->pendingAsyncMeshLoadCount());

  // The first render discovers the world meshes and starts their asynchronous imports.
  bool sceneReady = false;
  while (!app.quit) {
    app.processEvents();
    if (app.quit) break;

    if (sceneReady) scene.advance(kStepsPerFrame);

    suspension->update();
    app.beginFrame();
    app.renderViewer(*viewer);
    const bool wasLoading = loading.visible();
    loading.update(viewer->pendingAsyncMeshLoadCount());
    sceneReady = !loading.visible();
    if (wasLoading && !loading.visible())
      std::printf("assets: %zu meshes ready; simulation time=%.6f s\n", loading.total, scene.world->getWorldTime());
    loading.draw();
    app.endFrame();
  }

  const bool finite = scene.report();
  suspension.reset();
  viewer.reset();
  app.shutdown();
  return finite ? 0 : 1;
}
