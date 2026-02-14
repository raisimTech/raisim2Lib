#include <cmath>
#include <memory>

#include "rayrai/example_common.hpp"
#include "raisim/World.hpp"

int main(int argc, char* argv[]) {
  ExampleApp app;
  if (!app.init("rayrai_example_instancing", 1280, 720))
    return -1;

  auto world = std::make_shared<raisim::World>();
  world->addGround();

  auto viewer = std::make_shared<raisin::RayraiWindow>(world, 1280, 720);
  viewer->setBackgroundColor({60, 60, 70, 255});

  auto instancedBoxes = viewer->addInstancedVisuals("instanced_boxes", raisim::Shape::Box,
    glm::vec3(0.1f, 0.1f, 0.1f), glm::vec4(1.f, 0.2f, 0.2f, 1.f), glm::vec4(0.2f, 0.2f, 1.f, 1.f));
  instancedBoxes->setDetectable(true);

  const int gridSize = 300;
  const float spacing = 0.12f;
  for (int x = 0; x < gridSize; ++x) {
    for (int y = 0; y < gridSize; ++y) {
      const float weight = gridSize > 1 ? float(x) / float(gridSize - 1) : 0.f;
      instancedBoxes->addInstance(
        glm::vec3(1.0f + x * spacing, -2.5f + y * spacing, 0.15f), weight);
    }
  }

  while (!app.quit) {
    app.processEvents();
    if (app.quit)
      break;

    world->integrate();
    const double t = world->getWorldTime();

    if (instancedBoxes && instancedBoxes->count() > 0) {
      instancedBoxes->setPosition(0, glm::vec3(1.0f, -2.5f, 0.15f + 0.15f * std::sin(t * 3.0)));
    }

    app.beginFrame();
    app.renderViewer(*viewer);
    app.endFrame();
  }

  viewer.reset();
  app.shutdown();
  return 0;
}
