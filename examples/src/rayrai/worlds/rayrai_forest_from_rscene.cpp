// The rayrai_forest world, loaded from its .rscene file instead of being
// placed in C++.
#include "rayrai/example_common.hpp"
#include "rayrai/RsceneVisuals.hpp"
#include "example_resources.hpp"
#include "rayrai_loading_progress.hpp"

int main(int argc, char* argv[]) {
  auto world = std::make_shared<raisim::World>(
    exampleRscPath(argv[0], "forest/rayrai_forest.rscene"));
  ExampleApp app;
  if (!app.init("Raisim scene", 1280, 800)) return 1;
  {
    raisin::RayraiWindow viewer(world, 1280, 800);
    viewer.setAsyncMeshLoadingEnabled(true);
    raisin::applyRscene(*world->getRscene(), viewer);
    RayraiLoadingProgress loading("Loading forest assets");
    loading.update(viewer.pendingAsyncMeshLoadCount());
    while (!app.quit) {
      app.processEvents();
      // Physics starts once the meshes are loaded: eight 2 ms steps per frame.
      if (!loading.visible())
        for (int step = 0; step < 8; ++step) world->integrate();
      app.beginFrame();
      app.renderViewer(viewer);
      loading.update(viewer.pendingAsyncMeshLoadCount());
      loading.draw();
      app.endFrame();
    }
  } // Release graphics resources before closing the OpenGL context.
  app.shutdown();
  return 0;
}
