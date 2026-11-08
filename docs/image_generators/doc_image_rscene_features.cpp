// Generates small gallery images for .rscene record groups that are otherwise
// hard to understand from text alone. Each capture mirrors one group of
// records from docs/sections/RsceneFile.md: whole-scene environment records,
// resources, scene tree containers, lights/cameras, object/compound/wire
// records, particle-like records, terrain layers, and sensor/visual-aid
// records.

#include "doc_image_common.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <rayrai/Material.hpp>
#include <rayrai/SceneEffects.hpp>
#include <rayrai/Weather.hpp>

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;

std::filesystem::path findRepoAsset(const std::vector<std::filesystem::path>& candidates) {
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      return std::filesystem::canonical(candidate);
    }
  }
  return {};
}

void configureViewer(raisin::RayraiWindow& viewer,
                     raisin::RayraiWindow::RenderQualityPreset preset =
                         raisin::RayraiWindow::RenderQualityPreset::Balanced,
                     bool forceSimple = false) {
  auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(preset);
  quality.colorMode = forceSimple ? raisin::ViewerColorMode::FastLinear
                                  : raisin::ViewerColorMode::AcesApprox;
  quality.forceSimpleMaterialShading = forceSimple;
  quality.pbrToneMapping = !forceSimple;
  doc_image::applyCommonSceneOptions(quality, preset);
  quality.mainLightDirection = glm::normalize(glm::vec3(-0.35f, 0.45f, -1.0f));
  quality.mainLightDiffuse = glm::vec3(1.0f, 0.96f, 0.90f) * 1.55f;
  quality.mainLightAmbient = forceSimple ? glm::vec3(0.42f, 0.44f, 0.48f)
                                         : glm::vec3(0.30f, 0.32f, 0.36f);
  viewer.setRenderQualitySettings(quality);
}

bool captureFeature(raisin::RayraiWindow& viewer,
                    const std::filesystem::path& outputDir,
                    const char* fileName,
                    const glm::vec3& eye,
                    const glm::vec3& target,
                    float fov = 55.0f,
                    int supersample = 2) {
  doc_image::setCameraLookAt(viewer.getCamera(), eye, target, fov);
  raisin::RayraiWindow::RenderOverrides overrides;
  overrides.doShadows = true;
  const auto path = outputDir / fileName;
  if (!doc_image::captureScene(viewer, kWidth, kHeight, path, supersample, overrides)) {
    return false;
  }
  std::printf("doc_image: wrote %s\n", path.string().c_str());
  return true;
}

void setStatic(raisim::SingleBodyObject* object) {
  if (object) {
    object->setBodyType(raisim::BodyType::STATIC);
  }
}

void addCheckerProps(raisim::World& world) {
  world.addGround();

  auto* plinth = world.addBox(2.2, 1.2, 0.12, 1.0);
  plinth->setPosition(0.0, 0.0, 0.06);
  setStatic(plinth);
  plinth->setAppearance("0.48,0.49,0.50,1");

  auto* orange = world.addBox(0.45, 0.45, 0.45, 1.0);
  orange->setPosition(-0.65, -0.05, 0.345);
  setStatic(orange);
  orange->setAppearance("0.88,0.34,0.16,1");

  auto* blue = world.addCylinder(0.23, 0.65, 1.0);
  blue->setPosition(0.2, -0.05, 0.445);
  setStatic(blue);
  blue->setAppearance("0.12,0.42,0.85,1");

  auto* yellow = world.addSphere(0.24, 1.0);
  yellow->setPosition(0.82, 0.12, 0.36);
  setStatic(yellow);
  yellow->setAppearance("0.92,0.75,0.22,1");
}

bool renderEnvironmentWeather(const std::filesystem::path& outputDir) {
  auto world = std::make_shared<raisim::World>();
  raisin::RayraiWindow viewer(world, kWidth, kHeight);
  configureViewer(viewer, raisin::RayraiWindow::RenderQualityPreset::Fast, true);

  auto weather = raisin::RayraiWindow::defaultWeatherSettings(raisin::WeatherPreset::Overcast);
  weather.enabled = true;
  weather.timeOfDayHours = 17.5f;
  weather.affectSensors = false;
  weather.fogDensity = 0.035f;
  viewer.setWeatherSettings(weather);
  for (int i = 0; i < 45; ++i) {
    viewer.updateWeather(1.0 / 60.0);
  }

  addCheckerProps(*world);
  return captureFeature(viewer, outputDir, "rayrai_rscene_environment_weather.png",
                        glm::vec3(2.6f, -3.5f, 1.7f),
                        glm::vec3(0.05f, 0.0f, 0.35f));
}

bool renderResources(const std::filesystem::path& outputDir) {
  auto world = std::make_shared<raisim::World>();
  raisin::RayraiWindow viewer(world, kWidth, kHeight);
  configureViewer(viewer, raisin::RayraiWindow::RenderQualityPreset::Fast, true);

  world->addGround();

  auto metal = raisin::Material::simpleColor("polished resource material",
                                             glm::vec4(0.72f, 0.78f, 0.86f, 1.0f));
  auto rough = raisin::Material::simpleColor("rough resource material",
                                            glm::vec4(0.86f, 0.32f, 0.18f, 1.0f));

  auto metalSphere = viewer.addVisualSphere("material_metal", 0.34, glm::vec4(0.85f));
  metalSphere->setPosition(-1.05, -0.05, 0.34);
  metalSphere->setMaterialOverride(metal);
  metalSphere->setDetectable(true);

  auto roughBox = viewer.addVisualBox("material_rough", 0.55, 0.55, 0.55,
                                      glm::vec4(0.86f, 0.32f, 0.18f, 1.0f));
  roughBox->setPosition(-0.15, -0.05, 0.275);
  roughBox->setMaterialOverride(rough);
  roughBox->setDetectable(true);

  const auto markerPath = findRepoAsset({
      "rsc/aruco_marker/aruco_marker.dae",
      "../raisim/rsc/aruco_marker/aruco_marker.dae",
  });
  if (!markerPath.empty()) {
    auto marker = viewer.addVisualMesh("asset_marker", markerPath.string(),
                                       1.25, 1.25, 1.25,
                                       1.0f, 1.0f, 1.0f, 1.0f);
    marker->setPosition(0.75, -0.05, 0.05);
    marker->setUseMeshColor(true);
    marker->setTwoSided(true);
    marker->setDetectable(true);
  } else {
    auto fallback = viewer.addVisualCylinder("asset_marker_fallback", 0.28, 0.08,
                                             glm::vec4(0.1f, 0.1f, 0.1f, 1.0f));
    fallback->setPosition(0.75, -0.05, 0.08);
  }

  auto resourceStand = viewer.addVisualCapsule("articulated_resource_hint", 0.08, 0.9,
                                               glm::vec4(0.20f, 0.48f, 0.95f, 1.0f));
  resourceStand->setPosition(1.35, -0.05, 0.48);
  resourceStand->setDetectable(true);

  return captureFeature(viewer, outputDir, "rayrai_rscene_resources.png",
                        glm::vec3(2.5f, -3.1f, 1.45f),
                        glm::vec3(0.2f, -0.05f, 0.35f),
                        50.0f);
}

bool renderSceneTree(const std::filesystem::path& outputDir) {
  auto world = std::make_shared<raisim::World>();
  raisin::RayraiWindow viewer(world, kWidth, kHeight);
  configureViewer(viewer, raisin::RayraiWindow::RenderQualityPreset::Fast, true);
  world->addGround();

  for (int group = 0; group < 2; ++group) {
    const float x0 = group == 0 ? -0.95f : 0.95f;
    const glm::vec4 tint = group == 0 ? glm::vec4(0.20f, 0.55f, 0.95f, 0.22f)
                                      : glm::vec4(0.92f, 0.45f, 0.16f, 0.22f);
    auto volume = viewer.addVisualBox(group == 0 ? "group_volume_a" : "group_volume_b",
                                      1.25, 0.95, 0.72, tint);
    volume->setPosition(x0, 0.0, 0.38);
    volume->setTwoSided(true);
    volume->setDetectable(true);

    for (int i = 0; i < 3; ++i) {
      auto* box = world->addBox(0.28, 0.28, 0.28, 1.0);
      box->setPosition(x0 - 0.32 + 0.32 * i, -0.04 + 0.14 * (i % 2), 0.14);
      setStatic(box);
      box->setAppearance(group == 0 ? "0.15,0.42,0.80,1" : "0.86,0.34,0.18,1");
    }
  }

  return captureFeature(viewer, outputDir, "rayrai_rscene_scene_tree.png",
                        glm::vec3(2.4f, -3.0f, 1.45f),
                        glm::vec3(0.0f, 0.0f, 0.35f),
                        52.0f);
}

bool renderLightCamera(const std::filesystem::path& outputDir) {
  auto world = std::make_shared<raisim::World>();
  raisin::RayraiWindow viewer(world, kWidth, kHeight);
  configureViewer(viewer, raisin::RayraiWindow::RenderQualityPreset::Fast, true);
  addCheckerProps(*world);

  raisin::RayraiWindow::AdditionalLight warm;
  warm.type = raisin::LightType::POINT;
  warm.position = glm::vec3(-1.2f, -0.9f, 1.8f);
  warm.diffuse = glm::vec3(1.0f, 0.48f, 0.18f) * 1.8f;
  warm.specular = glm::vec3(0.25f, 0.12f, 0.05f);
  warm.linear = 0.12f;
  warm.quadratic = 0.05f;
  viewer.addAdditionalLight(warm);

  auto lightMarker = viewer.addVisualSphere("light_marker", 0.12,
                                            glm::vec4(1.0f, 0.78f, 0.15f, 1.0f));
  lightMarker->setPosition(warm.position);

  raisin::Camera sensorCam;
  doc_image::setCameraLookAt(sensorCam,
                             glm::vec3(1.5f, 0.95f, 1.0f),
                             glm::vec3(0.1f, -0.05f, 0.25f),
                             55.0f);
  sensorCam.farPlane = sensorCam.zFar = 2.2f;
  sensorCam.update(/*processKeyboard=*/false);
  auto frustum = viewer.addCameraFrustum("camera_record_frustum",
                                         glm::vec4(0.20f, 0.85f, 1.0f, 0.42f));
  frustum->updateFromCamera(sensorCam);
  frustum->setDetectable(true);

  return captureFeature(viewer, outputDir, "rayrai_rscene_light_camera.png",
                        glm::vec3(2.7f, -3.6f, 1.7f),
                        glm::vec3(0.05f, 0.0f, 0.45f),
                        56.0f);
}

bool renderObjectCompoundWire(const std::filesystem::path& outputDir) {
  auto world = std::make_shared<raisim::World>();
  raisin::RayraiWindow viewer(world, kWidth, kHeight);
  configureViewer(viewer, raisin::RayraiWindow::RenderQualityPreset::Fast, true);
  world->addGround();

  auto* box = world->addBox(0.42, 0.42, 0.42, 1.0);
  box->setPosition(-1.25, 0.2, 0.21);
  setStatic(box);
  box->setAppearance("0.86,0.34,0.18,1");

  auto* sphere = world->addSphere(0.24, 1.0);
  sphere->setPosition(-0.7, 0.2, 0.24);
  setStatic(sphere);
  sphere->setAppearance("0.92,0.76,0.20,1");

  auto* cylinder = world->addCylinder(0.18, 0.55, 1.0);
  cylinder->setPosition(-0.16, 0.2, 0.275);
  setStatic(cylinder);
  cylinder->setAppearance("0.18,0.46,0.74,1");

  // Compound body shown as one base with child shapes arranged around it.
  auto* compoundBase = world->addBox(0.55, 0.35, 0.25, 1.0);
  compoundBase->setPosition(0.7, 0.12, 0.125);
  setStatic(compoundBase);
  compoundBase->setAppearance("0.42,0.44,0.48,1");

  auto* childA = world->addBox(0.2, 0.6, 0.2, 1.0);
  childA->setPosition(0.7, 0.12, 0.36);
  setStatic(childA);
  childA->setAppearance("0.22,0.62,0.92,1");

  auto* childB = world->addSphere(0.18, 1.0);
  childB->setPosition(1.05, 0.12, 0.36);
  setStatic(childB);
  childB->setAppearance("0.95,0.55,0.20,1");

  // Wire constraint sketch: an anchor, a taut vertical wire, and a bob.
  auto anchor = viewer.addVisualBox("wire_anchor", 0.42, 0.14, 0.12,
                                    glm::vec4(0.12f, 0.12f, 0.14f, 1.0f));
  anchor->setPosition(1.65, -0.15, 1.25);
  auto wire = viewer.addVisualCylinder("wire_segment", 0.018, 1.0,
                                       glm::vec4(0.08f, 0.08f, 0.09f, 1.0f));
  wire->setPosition(1.65, -0.15, 0.78);
  auto* bob = world->addSphere(0.18, 1.0);
  bob->setPosition(1.65, -0.15, 0.24);
  setStatic(bob);
  bob->setAppearance("0.82,0.22,0.20,1");

  return captureFeature(viewer, outputDir, "rayrai_rscene_object_compound_wire.png",
                        glm::vec3(2.9f, -3.2f, 1.55f),
                        glm::vec3(0.15f, 0.05f, 0.42f),
                        54.0f);
}

bool renderParticles(const std::filesystem::path& outputDir) {
  auto world = std::make_shared<raisim::World>();
  raisin::RayraiWindow viewer(world, kWidth, kHeight);
  configureViewer(viewer, raisin::RayraiWindow::RenderQualityPreset::Fast, true);
  world->addGround();

  auto cloth = viewer.addInstancedVisuals(
      "deformable_sheet_nodes", raisim::Shape::Sphere, glm::vec3(0.055f),
      glm::vec4(0.1f, 0.65f, 1.0f, 1.0f),
      glm::vec4(0.65f, 0.90f, 1.0f, 1.0f));
  cloth->setDetectable(true);
  for (int y = 0; y < 7; ++y) {
    for (int x = 0; x < 9; ++x) {
      const float fx = -1.35f + 0.17f * static_cast<float>(x);
      const float fy = -0.45f + 0.17f * static_cast<float>(y);
      const float wave = 0.08f * std::sin(0.7f * static_cast<float>(x + y));
      cloth->addInstance(glm::vec3(fx, fy, 0.58f + wave),
                         static_cast<float>(x) / 8.0f);
    }
  }

  auto grains = viewer.addInstancedVisuals(
      "granular_particles", raisim::Shape::Sphere, glm::vec3(0.06f),
      glm::vec4(0.95f, 0.70f, 0.28f, 1.0f),
      glm::vec4(0.70f, 0.46f, 0.20f, 1.0f));
  grains->setDetectable(true);
  for (int layer = 0; layer < 5; ++layer) {
    const int count = 9 - layer;
    for (int i = 0; i < count; ++i) {
      const float x = 0.45f + 0.12f * static_cast<float>(i) -
                      0.06f * static_cast<float>(count - 1);
      const float y = 0.1f + 0.10f * static_cast<float>((i + layer) % 3);
      const float z = 0.07f + 0.105f * static_cast<float>(layer);
      grains->addInstance(glm::vec3(x, y, z), static_cast<float>(layer) / 4.0f);
    }
  }

  auto* bin = world->addBox(1.4, 0.08, 0.18, 1.0);
  bin->setPosition(0.85, 0.45, 0.09);
  setStatic(bin);
  bin->setAppearance("0.40,0.42,0.45,1");

  return captureFeature(viewer, outputDir, "rayrai_rscene_particles.png",
                        glm::vec3(2.4f, -2.9f, 1.45f),
                        glm::vec3(0.0f, 0.0f, 0.35f),
                        52.0f);
}

bool renderTerrainLayers(const std::filesystem::path& outputDir) {
  auto world = std::make_shared<raisim::World>();
  raisin::RayraiWindow viewer(world, kWidth, kHeight);
  configureViewer(viewer, raisin::RayraiWindow::RenderQualityPreset::Fast, true);
  world->addGround();

  auto sand = viewer.addVisualBox("terrain_splat_sand", 1.7, 1.05, 0.025,
                                  glm::vec4(0.82f, 0.68f, 0.36f, 0.86f));
  sand->setPosition(-0.45, -0.05, 0.018);
  sand->setTwoSided(true);
  auto grassPatch = viewer.addVisualBox("terrain_splat_grass", 1.45, 1.0, 0.03,
                                        glm::vec4(0.22f, 0.50f, 0.20f, 0.84f));
  grassPatch->setPosition(0.62, 0.18, 0.032);
  grassPatch->setTwoSided(true);

  for (int i = 0; i < 5; ++i) {
    auto* step = world->addBox(0.38, 0.38, 0.12 + 0.06 * i, 1.0);
    step->setPosition(-1.05f + 0.28f * i, 0.66f, 0.06f + 0.03f * i);
    setStatic(step);
    step->setAppearance("0.42,0.43,0.40,1");
  }

  auto foliage = viewer.addInstancedVisuals(
      "terrain_foliage_layer", raisim::Shape::Cylinder, glm::vec3(0.025f, 0.025f, 0.28f),
      glm::vec4(0.12f, 0.44f, 0.14f, 1.0f),
      glm::vec4(0.42f, 0.72f, 0.22f, 1.0f));
  foliage->setDetectable(true);
  for (int i = 0; i < 80; ++i) {
    const float x = 0.05f + 0.12f * static_cast<float>(i % 12);
    const float y = -0.28f + 0.13f * static_cast<float>(i / 12);
    foliage->addInstance(glm::vec3(x, y, 0.16f), static_cast<float>(i % 9) / 8.0f);
  }

  auto* boulder = world->addSphere(0.24, 1.0);
  boulder->setPosition(-0.75, -0.38, 0.24);
  setStatic(boulder);
  boulder->setAppearance("0.35,0.36,0.35,1");

  return captureFeature(viewer, outputDir, "rayrai_rscene_terrain_layers.png",
                        glm::vec3(2.35f, -2.95f, 1.55f),
                        glm::vec3(0.05f, 0.1f, 0.25f),
                        53.0f);
}

bool renderVisualAids(const std::filesystem::path& outputDir) {
  auto world = std::make_shared<raisim::World>();
  raisin::RayraiWindow viewer(world, kWidth, kHeight);
  configureViewer(viewer, raisin::RayraiWindow::RenderQualityPreset::Fast, true);
  world->addGround();

  auto* target = world->addBox(0.55, 0.55, 0.55, 1.0);
  target->setPosition(0.0, 0.0, 0.275);
  setStatic(target);
  target->setAppearance("0.64,0.66,0.68,1");

  raisin::Camera sensorCam;
  doc_image::setCameraLookAt(sensorCam,
                             glm::vec3(-1.45f, -0.85f, 0.95f),
                             glm::vec3(0.0f, 0.0f, 0.25f),
                             50.0f);
  sensorCam.farPlane = sensorCam.zFar = 2.1f;
  sensorCam.update(/*processKeyboard=*/false);
  auto frustum = viewer.addCameraFrustum("sensor_frustum",
                                         glm::vec4(0.25f, 0.85f, 1.0f, 0.40f));
  frustum->updateFromCamera(sensorCam);
  frustum->setDetectable(true);

  auto cloud = viewer.addPointCloud("point_cloud_record");
  cloud->setDetectable(true);
  cloud->pointSize = 5.0f;
  for (int i = 0; i < 180; ++i) {
    const float t = 0.12f * static_cast<float>(i);
    const float r = 0.35f + 0.002f * static_cast<float>(i);
    cloud->positions.emplace_back(0.85f + r * std::cos(t),
                                  0.25f + r * std::sin(t),
                                  0.18f + 0.0035f * static_cast<float>(i));
    cloud->colors.emplace_back(0.25f, 0.75f,
                               1.0f - 0.35f * static_cast<float>(i) / 180.0f,
                               1.0f);
  }
  cloud->updatePointBuffer();

  auto markers = viewer.addInstancedVisuals(
      "instanced_visual_record", raisim::Shape::Box, glm::vec3(0.12f, 0.12f, 0.12f),
      glm::vec4(0.92f, 0.34f, 0.18f, 1.0f),
      glm::vec4(0.95f, 0.78f, 0.22f, 1.0f));
  markers->setDetectable(true);
  for (int i = 0; i < 24; ++i) {
    markers->addInstance(glm::vec3(-0.85f + 0.18f * static_cast<float>(i % 6),
                                   0.78f + 0.18f * static_cast<float>(i / 6),
                                   0.08f),
                         static_cast<float>(i % 6) / 5.0f);
  }

  raisin::IrradianceVolume volume;
  volume.center = glm::vec3(0.08f, -0.42f, 0.55f);
  volume.halfExtents = glm::vec3(0.65f, 0.45f, 0.48f);
  volume.color = glm::vec3(0.45f, 0.62f, 1.0f);
  volume.strength = 0.75f;
  viewer.addIrradianceVolume(volume);
  auto volumeBox = viewer.addVisualBox("irradiance_volume_marker",
                                       1.3, 0.9, 0.96,
                                       glm::vec4(0.28f, 0.48f, 1.0f, 0.16f));
  volumeBox->setPosition(volume.center);
  volumeBox->setTwoSided(true);

  raisin::RayraiWindow::ReflectionProbe probe;
  probe.position = glm::vec3(1.2f, -0.65f, 0.55f);
  probe.radius = 1.2f;
  probe.strength = 0.8f;
  viewer.addReflectionProbe(probe);
  auto probeMarker = viewer.addVisualSphere("reflection_probe_marker", 0.18,
                                            glm::vec4(0.40f, 0.95f, 1.0f, 0.65f));
  probeMarker->setPosition(probe.position);
  probeMarker->setTwoSided(true);

  return captureFeature(viewer, outputDir, "rayrai_rscene_visual_aids.png",
                        glm::vec3(2.65f, -3.1f, 1.65f),
                        glm::vec3(0.15f, 0.1f, 0.45f),
                        55.0f);
}

}  // namespace

int main(int argc, char** argv) {
  const auto outputDir = doc_image::resolveOutputDir(argc, argv);
  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_rscene_features")) doc_image::finishAndExit(1);

  const bool ok =
      renderEnvironmentWeather(outputDir) &&
      renderResources(outputDir) &&
      renderSceneTree(outputDir) &&
      renderLightCamera(outputDir) &&
      renderObjectCompoundWire(outputDir) &&
      renderParticles(outputDir) &&
      renderTerrainLayers(outputDir) &&
      renderVisualAids(outputDir);

  doc_image::finishAndExit(ok ? 0 : 1);
}
