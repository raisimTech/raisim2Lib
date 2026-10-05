#include <iostream>
#include "forest_viewer.hpp"
#include "rayrai_loading_progress.hpp"

int main(int argc, char** argv) {
  try {
    std::filesystem::path assets = FOREST_ASSET_DIR;
    if (argc == 3 && std::string(argv[1]) == "--assets") assets = argv[2];
    else if (argc != 1) throw std::runtime_error("Usage: rayrai_forest [--assets DIR]");
    ExampleApp app;
    if (!app.init("Raisim forest - hills and physics",1280,800)) return 1;
    ImGui::GetIO().IniFilename = nullptr;
    {
      ForestViewer forest(assets,1280,800);
      RayraiLoadingProgress loading("Loading forest assets");
      loading.update(forest.viewer->pendingAsyncMeshLoadCount());
      while (!app.quit) {
        app.processEvents();
        if (app.quit) break;
        if (!loading.visible())
          for (int step=0;step<8;++step) forest.world->integrate();
        app.beginFrame();
        app.renderViewer(*forest.viewer);
        loading.update(forest.viewer->pendingAsyncMeshLoadCount());
        loading.draw();
        app.endFrame();
      }
    } // Release graphics resources before closing the OpenGL context.
    app.shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
