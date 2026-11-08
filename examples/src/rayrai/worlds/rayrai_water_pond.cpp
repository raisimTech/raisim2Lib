// A pond in a meadow: rayrai water with grass and trees on its banks and the low sun
// reflected on its surface.
#include <chrono>
#include <filesystem>
#include <iostream>

#include "example_resources.hpp"
#include "pond_viewer.hpp"
#include "rayrai_loading_progress.hpp"

int main(int argc, char* argv[]) {
  (void)argc;
  try {
    ExampleApp app;
    if (!app.init("Raisim pond - water rendering", 1280, 800)) return 1;
    ImGui::GetIO().IniFilename = nullptr;
    {
      PondViewer pond(std::filesystem::path(exampleRscPath(argv[0], "forest")).parent_path(), 1280, 800);
      RayraiLoadingProgress loading("Loading pond assets");
      loading.update(pond.viewer->pendingAsyncMeshLoadCount());
      auto last = std::chrono::steady_clock::now();
      while (!app.quit) {
        app.processEvents();
        const auto now = std::chrono::steady_clock::now();
        pond.viewer->advanceWaterTime(std::chrono::duration<double>(now - last).count());
        last = now;
        app.beginFrame();
        app.renderViewer(*pond.viewer);
        loading.update(pond.viewer->pendingAsyncMeshLoadCount());
        loading.draw();
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
