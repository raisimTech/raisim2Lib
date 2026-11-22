// Copyright (c) 2026 Raion Robotics Inc. All rights reserved.
#include <algorithm>
#include <cmath>
#include <string>

#include "glass_geometry.hpp"
#include "rayrai/Visuals.hpp"
#include "rayrai/example_common.hpp"

namespace {
using namespace raisin;
void run(ExampleApp& app) {
  raisim::World world;
  RayraiWindow viewer(world, 1280, 800, RayraiWindow::ThreadingMode::SingleThread);
  auto q = RenderQualitySettings::preset(RenderQualityPreset::High);
  q.highFidelityPbr = true;
  q.geometryRefraction = true;
  q.screenSpaceRefraction = true;
  q.geometryRefractionSamples = 10;
  q.geometryRefractionMaxBounces = 10;
  q.geometryRefractionMaxSamples = 65536;
  q.viewerMsaaSamples = 1;
  q.temporalAaEnabled = false;
  q.fxaaEnabled = true;
  q.autoExposureEnabled = false;
  q.screenSpaceAoEnabled = false;
  q.reflectiveGround = false;
  q.shadowsEnabled = true;
  q.addViewerFillLights = false;
  q.mainLightAmbient = glm::vec3(.09f, .11f, .14f);
  q.mainLightDiffuse = glm::vec3(2.7f);
  q.mainLightDirection = glm::normalize(glm::vec3(-.4, .2, -1));
  q.bloomEnabled = false;
  q.depthOfFieldEnabled = false;
  q.fogDensity = 0;
  q.heightFogEnabled = false;
  q.proceduralSkyBackgroundEnabled = false;
  q.pbrExposure = 1;
  q.gamma = 2.2f;
  viewer.setRenderQualitySettings(q);
  viewer.setLinearHdrRenderingEnabled(true);
  viewer.clearAdditionalLights();
  const auto box = [&](const std::string& name, glm::vec3 p, glm::vec3 size, glm::vec3 c, bool emissive = false) {
    auto v = viewer.addVisualBox(name, size.x, size.y, size.z, glm::vec4(1));
    v->setPosition(p.x, p.y, p.z);
    auto m = Material::pbr(name, glm::vec4(c, 1), 0, .8f);
    m.unlit = emissive;
    m.disableFog = true;
    v->setMaterialOverride(m);
    v->setDetectable(true);
    return v;
  };
  box("floor", {0, 0, -.15}, {16, 15, .2}, {.12, .15, .2});
  box("backdrop", {0, 3.7, 2.3}, {15, .15, 5}, {.7, .72, .76});
  for (int i = 0; i < 30; ++i)
    box("stripe" + std::to_string(i), {-7.25f + .5f * i, 3.59, 2}, {.18, .04, 4}, i % 3 == 0 ? glm::vec3(.04, .22, .32) : i % 3 == 1 ? glm::vec3(.7, .2, .035) : glm::vec3(.055, .065, .085));
  // Real offscreen emissive geometry: refraction and internal reflection rays
  // can hit these softboxes even though the primary camera cannot see them.
  box("softbox left", {-5, -6, 4.8}, {.15, 2.5, 2.2}, {9.5, 9.1, 8.6}, true)->setCastsShadows(false);
  box("softbox right", {5, -6, 4}, {.15, 2.5, 2}, {4, 6, 9}, true)->setCastsShadows(false);
  box("softbox ceiling", {0, 1, 6}, {6, 3, .1}, {4.5, 4.5, 4.5}, true)->setCastsShadows(false);
  auto clear = Material::glass("clear solid", 1, glm::vec3(.96, .99, 1), 4, 0, 1.5f);
  clear.dielectricPriority = 20;
  auto water = Material::glass("water", 1, glm::vec3(.62, .91, .94), 1.5, 0, 1.333f);
  water.dielectricPriority = 10;
  auto air = Material::glass("air bubble", 1, glm::vec3(1), 1, 0, 1);
  air.dielectricPriority = 30;
  const auto mesh = [&](const std::string& name, const glass_example::Geometry& g, glm::vec3 p, Material material) {
    auto v = viewer.addVisualCustomMesh(name, g.meshes(), glm::vec4(1));
    v->setPosition(p.x, p.y, p.z);
    v->setMaterialOverride(material);
    v->setDetectable(true);
    v->setCastsShadows(false);
    return v;
  };
  const auto sphere = [&](const std::string& name, glm::vec3 p, float radius, Material material) {
    auto v = viewer.addVisualSphere(name, radius, glm::vec4(1));
    v->setPosition(p.x, p.y, p.z);
    v->setMaterialOverride(material);
    v->setDetectable(true);
    v->setCastsShadows(false);
  };
  mesh("water glass", glass_example::cup(), {-3, 0, 0}, clear);
  // Water deliberately overlaps the inner wall and bottom; glass priority
  // wins there, producing glass/water rather than two air interfaces.
  mesh("water", glass_example::lathe({{0, .06f}, {.74f, .06f}, {.74f, 1.32f}, {0, 1.32f}}), {-3, 0, 0}, water);
  sphere("bubble", {-2.9, -.12, .85}, .18f, air);
  mesh("empty hollow cup", glass_example::cup(), {0, .7, 0}, clear);
  auto amber = clear;
  amber.volumeAttenuationColor = {.94, .55, .16};
  amber.volumeAttenuationDistance = .65f;
  auto torus = mesh("concave amber torus", glass_example::torus(), {3, .3, 1.1}, amber);
  torus->setOrientation(.70710678118, .70710678118, 0, 0);
  auto blue = clear;
  blue.volumeAttenuationColor = {.25, .7, .96};
  blue.volumeAttenuationDistance = 1;
  blue.dielectricPriority = 5;
  sphere("overlap blue", {-.55, -2, .85}, .8, blue);
  auto pink = clear;
  pink.volumeAttenuationColor = {.98, .48, .6};
  pink.volumeAttenuationDistance = 1;
  pink.ior = 1.38f;
  pink.dielectricPriority = 15;
  sphere("overlap pink", {.55, -1.8, .85}, .8, pink);
  viewer.updateObjectLists();
  const glm::vec3 eye(1.6f, -8.3f, 3.22f), lookAt(0, .4f, .9f);
  auto& camera = viewer.getCamera();
  camera.position = eye;
  camera.front = glm::normalize(lookAt - eye);
  camera.up = camera.worldUp = {0, 0, 1};
  camera.yaw = glm::degrees(std::atan2(camera.front.y, camera.front.x));
  camera.pitch = glm::degrees(std::asin(std::clamp(camera.front.z, -1.f, 1.f)));
  camera.target = lookAt;
  camera.zoom = 42;
  camera.movementSpeed = 3.f;
  camera.nearPlane = camera.zNear = .05f;
  camera.farPlane = camera.zFar = 100;
  camera.setGamma(q.gamma);
  camera.setBackgroundColorRgb255({15, 20, 30, 255});
  camera.setCameraFixedTarget(false);
  camera.setCameraFixedDistance(false);
  while (!app.quit) {
    app.processEvents();
    app.beginFrame();
    app.renderViewer(viewer);
    ImGui::Begin("Geometry refraction");
    ImGui::TextUnformatted("Water + air bubble / hollow cup / concave torus\nFront: overlapping glass solids");
    bool changed = ImGui::Checkbox("Geometry tracing", &q.geometryRefraction);
    int backend = int(q.geometryRefractionBackend);
    if (ImGui::Combo("Tracing backend", &backend, "Automatic\0Portable OpenGL\0Vulkan ray query (required)\0")) {
      q.geometryRefractionBackend = GeometryRefractionBackend(backend);
      changed = true;
    }
    changed |= ImGui::SliderInt("Paths / frame", &q.geometryRefractionSamples, 1, 32);
    changed |= ImGui::SliderInt("Max bounces", &q.geometryRefractionMaxBounces, 1, 128);
    changed |= ImGui::Checkbox("Low-discrepancy samples (experimental)", &q.geometryRefractionLowDiscrepancy);
    changed |= ImGui::Checkbox("Adaptive samples (experimental)", &q.geometryRefractionAdaptive);
    if (changed) viewer.setRenderQualitySettings(q);
    const auto d = viewer.geometryRefractionDiagnostics();
    ImGui::TextWrapped("%s", d.backendStatus.c_str());
    ImGui::Text("%s: %d samples | %zu triangles", q.geometryRefractionAdaptive ? "Scheduled budget" : "Accumulated", d.accumulatedSamples, d.triangles);
    if (const auto* focus = viewer.getTargetVisual())
      ImGui::Text("Orbiting: %s", focus->getName().c_str());
    else
      ImGui::TextUnformatted("Free fly: WASD + Space, drag to look, click an object to orbit");
    ImGui::TextUnformatted("Pause the camera to converge. No caustic illumination of the floor.");
    ImGui::End();
    app.endFrame();
  }
}
}  // namespace
int main() {
  ExampleApp app;
  if (!app.init("Rayrai nested glass", 1280, 800)) return 1;
  SDL_GL_SetSwapInterval(0);
  run(app);
  app.shutdown();
  return 0;
}
