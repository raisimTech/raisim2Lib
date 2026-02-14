#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "rayrai/example_common.hpp"
#include "raisim/World.hpp"

int main(int argc, char* argv[]) {
  ExampleApp app;
  if (!app.init("rayrai_example_pointcloud", 1280, 720))
    return -1;

  auto world = std::make_shared<raisim::World>();
  world->addGround();

  auto viewer = std::make_shared<raisin::RayraiWindow>(world, 1280, 720);
  viewer->setBackgroundColor({35, 35, 45, 255});

  auto cloud = viewer->addPointCloud("orbit_cloud");
  if (cloud) {
    cloud->pointSize = 3.0f;
    cloud->setDetectable(true);
  }

  constexpr float kPi = 3.14159265358979323846f;
  const int pointsPerRing = 360;
  const int ringCount = 6;

  while (!app.quit) {
    app.processEvents();
    if (app.quit)
      break;

    world->integrate();
    const double t = world->getWorldTime();

    if (cloud) {
      cloud->positions.clear();
      cloud->colors.clear();
      cloud->positions.reserve(pointsPerRing * ringCount);
      cloud->colors.reserve(pointsPerRing * ringCount);

      for (int r = 0; r < ringCount; ++r) {
        const float z = 0.2f + 0.08f * r;
        const float radius = 0.8f + 0.15f * r;
        const float hue = float(r) / float(std::max(1, ringCount - 1));
        const glm::vec4 color(0.2f + 0.8f * hue, 1.0f - 0.6f * hue, 0.3f, 1.0f);

        for (int i = 0; i < pointsPerRing; ++i) {
          const float angle = float(i) * 2.0f * kPi / float(pointsPerRing) + float(t);
          cloud->positions.emplace_back(
            glm::vec3(std::cos(angle) * radius, std::sin(angle) * radius, z));
          cloud->colors.emplace_back(color);
        }
      }

      cloud->updatePointBuffer();
    }

    app.beginFrame();
    app.renderViewer(*viewer);
    app.endFrame();
  }

  viewer.reset();
  app.shutdown();
  return 0;
}
