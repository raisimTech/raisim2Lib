// Benchmark sleeping islands with stacked box clusters.

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "rayrai_tcp_viewer_hint.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

int getIntFlag(int argc, char** argv, const std::string& prefix, int defaultValue) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind(prefix, 0) == 0) {
      return std::max(0, std::stoi(arg.substr(prefix.size())));
    }
  }
  return defaultValue;
}

void runBench(int steps, int hitStep) {
  raisim::World world;
  world.setTimeStep(0.002);
  world.addGround();

  world.setSleepingEnabled(true);
  world.setSleepingParameters(0.0015, 0.003, 2);

  constexpr int kIslands = 4;
  constexpr int kGrid = 3;
  const double spacing = 6.0;
  const double boxSize = 0.4;
  const double boxGap = boxSize * 1.1;
  const double baseHeight = 2.0;
  const double islandCenterOffset = (kGrid - 1) * boxGap * 0.5;

  std::vector<raisim::SingleBodyObject*> objects;
  objects.reserve(static_cast<size_t>(kIslands * kGrid * kGrid * kGrid));

  for (int island = 0; island < kIslands; ++island) {
    const double baseX = (island % 2) * spacing;
    const double baseY = (island / 2) * spacing;
    for (int ix = 0; ix < kGrid; ++ix) {
      for (int iy = 0; iy < kGrid; ++iy) {
        for (int iz = 0; iz < kGrid; ++iz) {
          auto* box = world.addBox(boxSize, boxSize, boxSize, 1.0);
          box->setAppearance("0.9, 0.2, 0.2, 1.0");
          const double x = baseX + ix * boxGap;
          const double y = baseY + iy * boxGap;
          const double z = baseHeight + iz * boxGap;
          box->setPosition(x, y, z);
          objects.push_back(box);
        }
      }
    }
  }

  const double sphereRadius = 1.0;
  auto* wakeSphere = world.addSphere(sphereRadius, 80.0);
  wakeSphere->setAppearance("1.0, 0.8, 0.1, 1.0");
  wakeSphere->setBodyType(raisim::BodyType::STATIC);
  wakeSphere->setPosition(0.0, 0.0, 50.0);

  std::vector<bool> lastSleeping(objects.size(), false);

  raisim::RaisimServer server(&world);
  server.launchServer();


  raisim_examples::warnIfNoClientConnected(server);
  const auto start = std::chrono::steady_clock::now();
  const int safeHitStep = (steps > 0) ? std::min(hitStep, steps - 1) : -1;
  for (int step = 0; step < steps; ++step) {
    if (step == safeHitStep) {
      const double targetX = islandCenterOffset;
      const double targetY = islandCenterOffset;
      const double targetZ = sphereRadius + 0.02;
      const double launchX = targetX - 8.0;
      wakeSphere->setBodyType(raisim::BodyType::DYNAMIC);
      wakeSphere->setPosition(launchX, targetY, targetZ);
      wakeSphere->setLinearVelocity(raisim::Vec<3>{8.0, 0.0, 0.0});
      wakeSphere->setAngularVelocity(raisim::Vec<3>{0.0, 0.0, 0.0});
      std::cout << "[sleep_bench] launching wake sphere at step " << step << "\n";
    }

    server.integrateWorldThreadSafe();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    for (size_t i = 0; i < objects.size(); ++i) {
      const bool sleeping = world.isObjectSleeping(objects[i]);
      if (sleeping != lastSleeping[i]) {
        objects[i]->setAppearance(sleeping ? "0.2, 0.6, 1.0, 1.0"
                                           : "0.9, 0.2, 0.2, 1.0");
        lastSleeping[i] = sleeping;
      }
    }
  }
  const auto end = std::chrono::steady_clock::now();

  server.killServer();

  const double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
  if (seconds > 0.0) {
    std::cout << "[sleep_bench] simulated " << steps << " steps in "
              << seconds << " s (" << (static_cast<double>(steps) / seconds)
              << " Hz)\n";
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);
  const std::string rscPath = (binaryPath.getDirectory() + "/../../rsc").getString();
  raisim::World::setActivationKey(rscPath + "/activation.raisim");

  const int steps = getIntFlag(argc, argv, "--steps=", 12000);
  const int defaultHitStep = 2500;  // 5s at 0.002s timestep.
  const int hitStep = getIntFlag(argc, argv, "--hit-step=", defaultHitStep);

  std::cout << "[sleep_bench] visualize on, steps=" << steps << std::endl;
  runBench(steps, hitStep);
  return 0;
}
