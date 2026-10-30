// Crates, a raft, barrels and beach balls ride the swell of the sea. The sea of the RaiSim world
// samples the same Gerstner waves that rayrai draws, at the physics time, so the water holds every
// body up where the drawn surface is: they heave on the crests and roll and pitch with the slope of
// the waves.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>

#include "fluid_waves_scene.hpp"
#include "rayrai/example_common.hpp"

int main() {
  try {
    ExampleApp app;
    if (!app.init("Raisim fluids - floating on waves", 1280, 800)) return 1;
    ImGui::GetIO().IniFilename = nullptr;
    {
      fluid_waves::Scene scene;
      std::shared_ptr<raisim::World> world(&scene.world(), [](raisim::World*) {});
      auto viewer = std::make_shared<raisin::RayraiWindow>(world, 1280, 800);
      auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(raisin::RayraiWindow::RenderQualityPreset::High);
      quality.gamma = 2.2f;
      quality.colorMode = raisin::ViewerColorMode::AcesApprox;
      quality.depthOfFieldEnabled = false;
      quality.reflectiveGround = false;
      quality.viewerMsaaSamples = 4;
      quality.addViewerFillLights = false;
      viewer->setRenderQualitySettings(quality);
      viewer->setWaterBody("sea", scene.water());

      raisin::RayraiWindow::WeatherSettings weather;
      weather.enabled = true;
      weather.preset = raisin::RayraiWindow::WeatherPreset::Clear;
      weather.useExplicitSunAngles = true;
      weather.sunAzimuthDegrees = 220.f;
      weather.sunElevationDegrees = 28.f;
      weather.cloudCoverage = .35f;
      viewer->setWeatherSettings(weather);
      auto& sun = viewer->getLight();
      sun.type = raisin::LightType::DIRECTIONAL;
      sun.setShadowsEnabled(true);
      sun.setShadowResolution(2048);
      const double az = 220.0 * fluid_waves::kPi / 180.0, el = 28.0 * fluid_waves::kPi / 180.0;
      sun.direction = glm::vec3(std::cos(az) * std::cos(el), std::sin(az) * std::cos(el), -std::sin(el));
      sun.diffuse = glm::vec3(1.f, .9f, .78f) * 12.f;
      sun.specular = glm::vec3(12.f);
      viewer->setShadowOrtho(20, .1f, 80);

      auto& camera = viewer->getCamera();
      camera.position = {-7.5f, -9.0f, 3.2f};
      camera.yaw = 50.f;
      camera.pitch = -16.f;
      camera.zoom = 50;

      auto last = std::chrono::steady_clock::now();
      double lag = 0.0;
      while (!app.quit) {
        app.processEvents();
        const auto now = std::chrono::steady_clock::now();
        lag += std::clamp(std::chrono::duration<double>(now - last).count(), 0.0, 0.05);
        last = now;
        for (; lag >= fluid_waves::kTimeStep; lag -= fluid_waves::kTimeStep) scene.Step();
        viewer->setWaterTimeSeconds(scene.world().getWorldTime());
        app.beginFrame();
        app.renderViewer(*viewer);
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
