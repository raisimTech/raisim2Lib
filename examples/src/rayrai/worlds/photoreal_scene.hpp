// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#pragma once

// A photoreal world loaded from a .rscene file with an ANYmal C standing in it, shared by
// rayrai_city and rayrai_warehouse. RaiSim builds the scene's colliders, props and loose bodies
// from the file; the .rscene reader creates no articulated systems, so the robot is added here.

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

#include "example_resources.hpp"
#include "rayrai/example_common.hpp"
#include "rayrai_loading_progress.hpp"

namespace photoreal_scene {

struct Scene {
  const char* program;  ///< Executable name, for the usage line.
  const char* title;    ///< Window title.
  const char* file;     ///< The .rscene file, relative to rsc.
  const char* name;     ///< "city": the loading overlay.
  double robotX, robotY, robotYaw;  ///< A spot the scene keeps free of props [m, rad].
};

inline raisim::ArticulatedSystem* addStandingAnymal(raisim::World& world, const std::string& urdf, const Scene& scene) {
  raisim::ArticulatedSystemOption options;
  options.convexifyCollisionMeshes = true;
  auto* anymal = world.addArticulatedSystem(urdf, "", {}, raisim::CollisionGroup(1), raisim::CollisionGroup(-1), options);
  anymal->setName("anymal");
  Eigen::VectorXd gc(anymal->getGeneralizedCoordinateDim()), gv(anymal->getDOF());
  gc << scene.robotX, scene.robotY, 0.56, std::cos(scene.robotYaw / 2), 0, 0, std::sin(scene.robotYaw / 2), 0.03, 0.4,
      -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
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

/// Shows the scene; --scene FILE loads another .rscene file.
inline int run(int argc, char* argv[], const Scene& scene) {
  std::filesystem::path file = exampleRscPath(argv[0], scene.file);
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--scene" && i + 1 < argc)
      file = argv[++i];
    else {
      std::cerr << "Usage: " << scene.program << " [--scene FILE.rscene]\n";
      return 2;
    }
  }
  auto world = std::make_shared<raisim::World>(file.string());
  addStandingAnymal(*world, exampleRscPath(argv[0], "anymal_c/urdf/anymal.urdf"), scene);

  ExampleApp app;
  if (!app.init(scene.title, 1280, 800)) return 1;
  ImGui::GetIO().IniFilename = nullptr;
  {
    // The window applies the world's scene.
    raisin::RayraiWindow viewer(world, 1280, 800);
    // Meshes load asynchronously by default, so this line changes nothing; it shows the
    // switch. setAsyncMeshLoadingEnabled(false) loads the scene's meshes before it returns.
    viewer.setAsyncMeshLoadingEnabled(true);
    RayraiLoadingProgress loading("Loading " + std::string(scene.name) + " assets");
    loading.update(viewer.pendingAsyncMeshLoadCount());
    while (!app.quit) {
      app.processEvents();
      if (app.quit) break;
      // Physics starts once the meshes are loaded: eight 2 ms steps per frame.
      if (!loading.visible())
        for (int step = 0; step < 8; ++step) world->integrate();
      app.beginFrame();
      app.renderViewer(viewer);
      loading.update(viewer.pendingAsyncMeshLoadCount());
      loading.draw();
      app.endFrame();
    }
  }  // Release graphics resources before closing the OpenGL context.
  app.shutdown();
  return 0;
}

}  // namespace photoreal_scene
