// A creek running down a hillside into a pool: rayrai's shallow-water simulation carries the water
// around rocks and over small drops, and wooden balls float down it on the simulated current.
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>

#include "example_resources.hpp"
#include "rayrai_loading_progress.hpp"
#include "stream_viewer.hpp"

int main(int argc, char* argv[]) {
  (void)argc;
  try {
    ExampleApp app;
    if (!app.init("Raisim creek - flowing water", 1280, 800)) return 1;
    ImGui::GetIO().IniFilename = nullptr;
    {
      StreamViewer creek(std::filesystem::path(exampleRscPath(argv[0], "forest")).parent_path(), 1280, 800);
      RayraiLoadingProgress loading("Loading creek assets");
      while (!app.quit && creek.viewer->pendingAsyncMeshLoadCount() > 0) {
        app.processEvents();
        app.beginFrame();
        app.renderViewer(*creek.viewer);
        loading.update(creek.viewer->pendingAsyncMeshLoadCount());
        loading.draw();
        app.endFrame();
      }
      creek.warmUp(60);
      auto last = std::chrono::steady_clock::now();
      while (!app.quit) {
        app.processEvents();
        const auto now = std::chrono::steady_clock::now();
        creek.step(std::clamp(std::chrono::duration<double>(now - last).count(), StreamViewer::kTimeStep, .05));
        last = now;
        app.beginFrame();
        app.renderViewer(*creek.viewer);
        app.endFrame();
      }
    }  // Release graphics resources before closing the OpenGL context.
    app.shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
