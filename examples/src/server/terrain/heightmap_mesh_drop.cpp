// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include <cmath>
#include <string>
#include <vector>

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"
#include "example_resources.hpp"

int main(int argc, char* argv[]) {

    constexpr size_t X_SAMPLES = 120;
    constexpr size_t Y_SAMPLES = 120;
    constexpr double X_SIZE = 20.0;
    constexpr double Y_SIZE = 20.0;
    /// the terrain is smaller than the sample terrain, so the elevation is scaled down as well
    constexpr double HEIGHT_SCALE = 0.4;

    raisim::World world;
    world.setTimeStep(0.0025);

    std::vector<double> terrainHeights(X_SAMPLES * Y_SAMPLES, 0.0);
    std::vector<raisim::ColorRGB> terrainColors(X_SAMPLES * Y_SAMPLES, {0, 0, 0});

    for (size_t i = 0; i < X_SAMPLES; i++) {
      for (size_t j = 0; j < Y_SAMPLES; j++) {
        const double x = static_cast<double>(i) / X_SAMPLES;
        const double y = static_cast<double>(j) / Y_SAMPLES;
        const double h = 4.0 * std::sin(3.0 * x + 0.8) + 3.0 * std::cos(5.0 * y + 1.1) +
                         2.2 * std::sin(2.0 * (x + y) + 0.2) +
                         1.4 * std::sin(11.0 * x + 2.3) * std::cos(9.0 * y + 0.7) +
                         0.8 * std::sin(23.0 * x + 1.9) * std::sin(19.0 * y + 3.1) +
                         0.5 * std::cos(37.0 * (x - y) + 0.5) +
                         0.3 * std::sin(61.0 * x + 0.9) * std::cos(53.0 * y + 2.1);

        terrainHeights[i * Y_SAMPLES + j] = HEIGHT_SCALE * h + 3.0;

        const uint8_t r = static_cast<uint8_t>(80 + 80 * (0.5 + 0.5 * std::sin(4.0 * x)));
        const uint8_t g = static_cast<uint8_t>(100 + 90 * (0.5 + 0.5 * std::cos(4.0 * y + 0.4)));
        const uint8_t b = static_cast<uint8_t>(140 + 70 * (0.5 + 0.5 * std::sin(3.0 * (x + y))));
        terrainColors[i * Y_SAMPLES + j] = {r, g, b};
      }
    }

    auto terrain = world.addHeightMap(X_SAMPLES, Y_SAMPLES, X_SIZE, Y_SIZE, 0.0, 0.0, terrainHeights);
    terrain->setName("drop_heightmap");
    terrain->setColor(terrainColors);

    /// the meshes that are dropped on the terrain
    const std::string meshPath = exampleRscPath(argv[0], "monkey/monkey.obj");

    raisim::RaisimServer server(&world);
    server.launchServer();
    raisim_examples::warnIfNoClientConnected(server);
    server.setCameraPositionAndLookAt({8., 8., 12.}, {0., 0., 4.});

    size_t dropped = 0;
    constexpr size_t targetDropCount = 20;
    constexpr size_t COLS = 5;
    constexpr double spacing = 2.5;
    constexpr double startX = -spacing * (COLS - 1) * 0.5;
    constexpr double startY = -spacing * ((targetDropCount / COLS) - 1) * 0.5;

    int i = 0;
    while (true) {
      RS_TIMED_LOOP(int(world.getTimeStep() * 1e6));
      server.integrateWorldThreadSafe([&]() {
        /// drop one mesh every 200 steps (0.5 s) so that they land one by one
        if (dropped < targetDropCount && (i % 200 == 0)) {
          const size_t row = dropped / COLS;
          const size_t col = dropped % COLS;
          const double x = startX + spacing * static_cast<double>(col);
          const double y = startY + spacing * static_cast<double>(row);
          const double z = 11.0;

          const double scale = 0.5 + 0.05 * static_cast<double>(dropped % 5);
          /// the monkey mesh is not 2-manifold, so CoACD decomposition fails on it;
          /// collide with the original triangle mesh directly
          auto* mesh = world.addMesh(meshPath, 2.0, scale, "",
                                     raisim::MeshCollisionMode::ORIGINAL_MESH);
          mesh->setName("mesh_" + std::to_string(dropped));
          mesh->setPosition(x, y, z);

          /// tilt each mesh differently so that they do not land identically
          const double angle = 0.31 * static_cast<double>(dropped);
          const double axisX = std::cos(0.7 * static_cast<double>(dropped));
          const double axisY = std::sin(0.7 * static_cast<double>(dropped));
          mesh->setOrientation(std::cos(0.5 * angle),
                               std::sin(0.5 * angle) * axisX,
                               std::sin(0.5 * angle) * axisY,
                               0.0);
          mesh->setVelocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
          mesh->setAppearance(std::to_string(0.3 + 0.15 * (dropped % 5)) + ", " +
                              std::to_string(0.3 + 0.15 * (dropped % 4)) + ", " +
                              std::to_string(0.3 + 0.15 * (dropped % 3)) + ", 1.0");
          ++dropped;
        }
        ++i;
      });
    }

    server.killServer();
    return 0;
}
