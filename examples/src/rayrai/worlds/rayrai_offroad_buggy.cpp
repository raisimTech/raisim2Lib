// An off-road buggy on a dirt race circuit, driven with the arrow keys.
//
// The car is one articulated system from rsc/offroad_buggy/buggy.urdf: four
// double-wishbone corners, coilovers, tie rods and a rack-and-pinion steering,
// written as a kinematic tree whose loops are closed by pin, equality and mimic
// constraints. The springs and dampers are joint springs and joint damping of
// the coilovers' sliding joints. The springs are drawn at their simulated
// length and coloured by their compression.
//
// The circuit runs through eroded hills with trees, boulders and grass
// (offroad_buggy_scene.hpp), with a start/finish gantry, pits, tyre walls,
// hay bales and cones that the car can knock over, and a spectator area
// (offroad_buggy_venue.hpp).
//
// --validate runs the physics checks of offroad_buggy_validation.hpp without a
// window; --screenshot PNG renders a frame of a short scripted drive.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <glbinding/gl/gl.h>
#include "example_resources.hpp"
#include "offroad_buggy_scene.hpp"
#include "offroad_buggy_validation.hpp"
#include "offroad_buggy_vehicle.hpp"
#include "offroad_buggy_visuals.hpp"
#include "offroad_buggy_venue.hpp"
#include "rayrai/Visuals.hpp"
#include "rayrai/example_common.hpp"
#include "rayrai_example_utils.hpp"
#include "rayrai_loading_progress.hpp"

namespace {

using Viewer = raisin::RayraiWindow;
constexpr double kTimeStep = 0.001;
constexpr const char* kCornerLabels[4] = {"Front left", "Front right", "Rear left", "Rear right"};

using offroad::springColour;
using offroad::toGlmQuat;

ImU32 imColour(const glm::vec3& c, float alpha = 1.f) { return ImGui::ColorConvertFloat4ToU32(ImVec4(c.r, c.g, c.b, alpha)); }

enum class CameraMode { Chase, Suspension, Side, Free };
/// Look-at point in the chassis frame, orbit distance, and yaw (relative to the heading) and pitch
/// of the view direction, per camera mode.
struct CameraView {
  Eigen::Vector3d focus;
  float distance, yaw, pitch;
};
const std::array<CameraView, 4> kCameraViews = {{{{0.0, 0.0, 0.55}, 7.5f, 0.f, -13.f},       // chase
                                                 {{-1.36, 0.80, 0.0}, 1.45f, -28.f, -10.f},  // rear-left corner from behind: coilover, arms, half shaft
                                                 {{0.0, 0.0, 0.0}, 6.5f, -90.f, -3.f},       // low side view: all four wheels articulating
                                                 {{0.0, 0.0, 0.55}, 7.5f, 0.f, -13.f}}};     // free orbit

/// Set the locked orbit distance of the camera.
void setOrbitDistance(raisin::Camera& camera, float distance) {
  const float yaw = glm::radians(camera.yaw), pitch = glm::radians(camera.pitch);
  const glm::vec3 front(std::cos(yaw) * std::cos(pitch), std::sin(yaw) * std::cos(pitch), std::sin(pitch));
  camera.setCameraFixedDistance(false);
  camera.position = camera.target - front * distance;
  camera.setCameraFixedDistance(true);
}
const char* cameraModeName(CameraMode m) {
  switch (m) {
    case CameraMode::Chase:
      return "chase";
    case CameraMode::Suspension:
      return "rear suspension";
    case CameraMode::Side:
      return "side";
    case CameraMode::Free:
      return "free orbit";
  }
  return "";
}

struct Options {
  std::filesystem::path screenshot;
  std::vector<float> view;  // fixed camera for captures: x y z yaw pitch
  int captureFrames = 600;
  int cameraMode = 0;
  bool hud = false;
  bool validate = false;
  double grassDensity = 1.0;
};

/// The whole window with its overlays, read from the back buffer before it is swapped.
bool saveWindowScreenshot(SDL_Window* window, const std::filesystem::path& path) {
  int width = 0, height = 0;
  SDL_GL_GetDrawableSize(window, &width, &height);
  const size_t stride = size_t(width) * 4;
  std::vector<unsigned char> rgba(stride * height), flipped(stride * height);
  gl::glBindFramebuffer(gl::GL_READ_FRAMEBUFFER, 0);
  gl::glReadBuffer(gl::GL_BACK);
  gl::glPixelStorei(gl::GL_PACK_ALIGNMENT, 1);
  gl::glReadPixels(0, 0, width, height, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
  for (int y = 0; y < height; ++y) std::memcpy(&flipped[y * stride], &rgba[(height - 1 - y) * stride], stride);
  return stbi_write_png(path.string().c_str(), width, height, 4, flipped.data(), int(stride)) != 0;
}

/// Direction of the sun in the sky HDR (rsc/city/sky), from the sun towards the scene, in rayrai's
/// mapping of an equirectangular image (azimuth atan2(y, x), zenith at the top row).
const glm::vec3 kSunDirection = glm::normalize(glm::vec3(.5556f, .3738f, -.7427f));

void configureRendering(Viewer& viewer, const std::filesystem::path& sky) {
  auto quality = Viewer::defaultRenderQualitySettings(Viewer::RenderQualityPreset::High);
  quality.gamma = 2.2f;
  quality.colorMode = raisin::ViewerColorMode::AcesApprox;
  quality.pbrExposure = 1.05f;
  quality.pbrEnvironmentLightingTint = glm::vec3(.65f);
  quality.depthOfFieldEnabled = false;
  quality.reflectiveGround = false;
  quality.viewerMsaaSamples = 4;
  quality.directionalShadowCascadeCount = 3;
  quality.directionalShadowCascadeMaxDistance = 90;
  quality.addViewerFillLights = false;
  // The full PBR shader blends the trail texture over the meadow (terrain splat layer).
  quality.highFidelityPbr = true;
  viewer.setRenderQualitySettings(quality);
  Viewer::WeatherSettings weather;
  weather.enabled = true;
  weather.preset = Viewer::WeatherPreset::Clear;
  weather.useExplicitSunAngles = true;
  weather.sunAzimuthDegrees = 214;  // as in the HDR
  weather.sunElevationDegrees = 48;
  weather.sunDiskSize = .001f;
  weather.windSpeed = 2.0f;
  weather.cloudCoverage = .12f;
  weather.visibilityMeters = 6000.f;  // light haze only: below 5 km rayrai adds height fog
  weather.fogColor = glm::vec3(.66f, .72f, .78f);
  viewer.setWeatherSettings(weather);
  // The weather sky lights the scene; the partly cloudy HDR sky of rsc/city is the background.
  viewer.generateWeatherSkyEnvironment(256, 32, false);
  viewer.setSceneEnvironmentBackground(viewer.acquireSceneHdrCubemapCached(sky.string(), 1024), 1.f);
  auto& sun = viewer.getLight();
  sun.type = raisin::LightType::DIRECTIONAL;
  sun.setShadowsEnabled(true);
  sun.setShadowResolution(2048);
  sun.setShadowParams(.0008f, 1.f, 1.25f);
  viewer.setShadowOrtho(45, .1f, 220);
  sun.direction = kSunDirection;
  sun.ambient = glm::vec3(.13f);
  sun.diffuse = glm::vec3(1, .95f, .86f) * 17.f;
  sun.specular = glm::vec3(17);
}

/// The terrain as chunks of the height grid, drawn with five ground materials: meadow, the packed
/// racing surface, loose dirt, gravel and rock. Each grid triangle is cut along the 0.5 contours of
/// the surface masks (marching triangles), first by rock, then gravel, then bare ground and then
/// loose dirt, and each piece is drawn with its material. Vertex colours add lush and dry patches
/// to the meadow, darken hollows, and darken the worn racing line.
///
/// This replaces rayrai's heightmap splat layer (setHeightmapSplatLayer), whose blend does not reach
/// the screen in rayrai 2.8.1: its samplers use texture units 39 and 40, above the 32 that a
/// fragment shader has on most GPUs, and the lean PBR shader that draws plain textured meshes
/// ignores it.
std::vector<std::shared_ptr<raisin::Visuals>> addTerrainVisuals(Viewer& viewer, const offroad::TerrainInfo& info, const std::vector<double>& heights, const std::vector<Eigen::Vector3f>& tint,
                                                                const offroad::Surface& surface, const std::filesystem::path& textures) {
  using raisin::Material;
  enum Ground { Meadow, Track, Loose, Gravel, Rock, kGrounds };
  struct Layer {
    const char* name;
    const char* texture;  // Poly Haven texture in rsc/offroad_buggy/terrain/textures
    double tile;          // metres per texture repeat
    glm::vec4 colour;
    float roughness, normalStrength;
  };
  const std::array<Layer, kGrounds> layers = {{{"meadow", "rocky_terrain_02", 7.0, glm::vec4(1.f), .95f, .8f},
                                               {"track", "dry_ground_rocks", 4.5, glm::vec4(.70f, .60f, .50f, 1.f), .92f, 1.f},
                                               {"loose dirt", "brown_mud_dry", 3.0, glm::vec4(.95f, .90f, .86f, 1.f), .96f, 1.f},
                                               {"gravel", "rocky_trail", 3.0, glm::vec4(.70f, .68f, .65f, 1.f), .9f, 1.f},
                                               {"rock", "rock_face_03", 6.0, glm::vec4(.9f, .88f, .85f, 1.f), .85f, 1.2f}}};
  std::array<Material, kGrounds> materials;
  for (int l = 0; l < kGrounds; ++l) {
    auto& m = materials[l] = Material::pbr(layers[l].name, layers[l].colour, 0.f, layers[l].roughness);
    const auto file = textures / layers[l].texture;
    m.albedoMap = Viewer::loadTextureWithTiling((file.string() + "_diff_2k.jpg").c_str(), true);
    m.normalMap = Viewer::loadTextureWithTiling((file.string() + "_nor_gl_1k.jpg").c_str(), false);
    m.normalStrength = layers[l].normalStrength;
    m.useVertexColorAsAlbedo = true;
  }
  const int n = info.samples, chunks = 8, per = (n - 1) / chunks;
  const double cell = info.extent / (n - 1), half = info.extent / 2;
  auto height = [&](int i, int j) { return heights[size_t(std::clamp(j, 0, n - 1)) * n + std::clamp(i, 0, n - 1)]; };
  // A point of the surface: grid coordinates (fractional on cut edges) and the mask values there.
  struct Point {
    double i, j;
    std::array<double, 4> mask;  // rock, gravel, bare, loose: the order of the cuts
  };
  struct Polygon {
    std::array<Point, 8> p;  // a triangle cut by four lines has at most seven corners
    int size = 0;
  };
  auto makeVertex = [&](int l, const Point& p) {
    const double x = -half + p.i * cell, y = -half + p.j * cell;
    const int i0 = std::min(int(p.i), n - 2), j0 = std::min(int(p.j), n - 2);
    const double fi = p.i - i0, fj = p.j - j0;
    auto lerp2 = [&](auto f) { return (1 - fj) * ((1 - fi) * f(i0, j0) + fi * f(i0 + 1, j0)) + fj * ((1 - fi) * f(i0, j0 + 1) + fi * f(i0 + 1, j0 + 1)); };
    Vertex v;
    v.Position = glm::vec3(float(x), float(y), float(lerp2(height)));
    const double dx = lerp2([&](int i, int j) { return (height(i + 1, j) - height(i - 1, j)) / (2 * cell); });
    const double dy = lerp2([&](int i, int j) { return (height(i, j + 1) - height(i, j - 1)) / (2 * cell); });
    v.Normal = glm::normalize(glm::vec3(float(-dx), float(-dy), 1.f));
    v.TexCoords = glm::vec2(float(x / layers[l].tile), float(y / layers[l].tile));
    const auto& c = tint[size_t(std::lround(p.j)) * n + size_t(std::lround(p.i))];
    // The bare grounds keep only the shading of the tint (darker in hollows), not its grass colour;
    // the racing line is darker where the tyres have worn and compacted it.
    float shade = std::min(c.y() / 0.93f, 1.f);
    if (l == Track) shade *= float(1.0 - 0.4 * surface.wear.at(x, y));
    v.Color = l == Meadow ? glm::vec4(c.x(), c.y(), c.z(), 1.f) : glm::vec4(shade, shade, shade, 1.f);
    return v;
  };
  std::vector<std::shared_ptr<raisin::Visuals>> visuals;
  for (int cy = 0; cy < chunks; ++cy)
    for (int cx = 0; cx < chunks; ++cx) {
      std::array<std::vector<Vertex>, kGrounds> vertices;
      std::array<std::vector<unsigned int>, kGrounds> indices;
      auto emit = [&](int l, const Polygon& poly) {
        if (poly.size < 3) return;
        const auto base = unsigned(vertices[l].size());
        for (int k = 0; k < poly.size; ++k) vertices[l].push_back(makeVertex(l, poly.p[k]));
        for (int k = 1; k + 1 < poly.size; ++k) indices[l].insert(indices[l].end(), {base, base + k, base + k + 1});
      };
      // Split a polygon along the 0.5 contour of one mask, interpolating the others on cut edges.
      auto split = [](const Polygon& poly, int m, Polygon& below, Polygon& above) {
        below.size = above.size = 0;
        for (int k = 0; k < poly.size; ++k) {
          const Point& a = poly.p[k];
          const Point& b = poly.p[(k + 1) % poly.size];
          Polygon& side = a.mask[m] > 0.5 ? above : below;
          side.p[side.size++] = a;
          if ((a.mask[m] > 0.5) != (b.mask[m] > 0.5)) {
            const double t = (0.5 - a.mask[m]) / (b.mask[m] - a.mask[m]);
            Point cut{a.i + t * (b.i - a.i), a.j + t * (b.j - a.j), {}};
            for (int q = 0; q < 4; ++q) cut.mask[q] = a.mask[q] + t * (b.mask[q] - a.mask[q]);
            below.p[below.size++] = cut;
            above.p[above.size++] = cut;
          }
        }
      };
      auto triangle = [&](const Point& a, const Point& b, const Point& c) {
        Polygon all, rest, rock, gravel, grass, bare, track, loose;
        all.p[0] = a;
        all.p[1] = b;
        all.p[2] = c;
        all.size = 3;
        split(all, 0, rest, rock);
        emit(Rock, rock);
        split(rest, 1, all, gravel);
        emit(Gravel, gravel);
        split(all, 2, grass, bare);
        emit(Meadow, grass);
        split(bare, 3, track, loose);
        emit(Track, track);
        emit(Loose, loose);
      };
      auto point = [&](int i, int j) {
        const double x = -half + i * cell, y = -half + j * cell;
        return Point{double(i), double(j), {surface.rock.at(x, y), surface.gravel.at(x, y), surface.bare.at(x, y), surface.loose.at(x, y)}};
      };
      for (int j = cy * per; j < (cy + 1) * per; ++j)
        for (int i = cx * per; i < (cx + 1) * per; ++i) {
          const Point a = point(i, j), b = point(i + 1, j), c = point(i, j + 1), d = point(i + 1, j + 1);
          triangle(a, b, d);
          triangle(a, d, c);
        }
      for (int l = 0; l < kGrounds; ++l) {
        if (indices[l].empty()) continue;
        auto meshes = std::make_shared<std::vector<std::shared_ptr<OpenGLMesh>>>();
        meshes->push_back(std::make_shared<OpenGLMesh>(std::move(vertices[l]), std::move(indices[l])));
        meshes->front()->material = materials[l];
        visuals.push_back(viewer.addVisualCustomMesh(std::string(layers[l].name) + " " + std::to_string(cy * chunks + cx), meshes, glm::vec4(1.f), false));
      }
    }
  return visuals;
}

/// Grass batches of one 32 m tile, and the tile's centre.
struct GrassTile {
  glm::vec2 centre;
  std::vector<std::shared_ptr<raisin::InstancedVisuals>> batches;
};

/// Draw only the grass tiles near the camera: tiles beyond the fade distance draw a single tuft
/// (rayrai has no hidden state for an instanced batch), and farther tiles every second or third tuft.
void updateGrassTiles(std::vector<GrassTile>& tiles, const glm::vec3& camera) {
  for (auto& tile : tiles) {
    const float distance = std::max(0.f, glm::length(tile.centre - glm::vec2(camera)) - 23.f);  // to the tile's rim
    for (auto& batch : tile.batches) {
      batch->setMaxRenderedInstances(distance > 70.f ? 1 : 0);
      batch->setRenderedInstanceStride(distance > 45.f ? 3 : distance > 20.f ? 2 : 1);
    }
  }
}

void addScenery(Viewer& viewer, const offroad::Scenery& scenery, const offroad::Venue& venue, const std::filesystem::path& rsc, std::vector<std::shared_ptr<raisin::InstancedVisuals>>& keep,
                std::vector<std::shared_ptr<raisin::Visuals>>& singles, std::vector<GrassTile>& grass) {
  const auto forest = rsc / "forest", props = rsc / "offroad_buggy" / "props";
  // tint: vary the colour of each instance a little (plants and rocks).
  auto instanced = [&](const std::string& name, const std::filesystem::path& mesh, const std::vector<offroad::Placement>& all, int type, bool foliage, float sway, int tile = -1, double tileSize = 0,
                       int tiles = 1, bool tint = true) {
    std::vector<raisin::InstancedVisuals::InstanceSpec> instances;
    for (const auto& p : all)
      if ((type < 0 || p.type == type) &&
          (tile < 0 || std::clamp(int((p.x + tiles * tileSize / 2) / tileSize), 0, tiles - 1) + tiles * std::clamp(int((p.y + tiles * tileSize / 2) / tileSize), 0, tiles - 1) == tile)) {
        raisin::InstancedVisuals::InstanceSpec instance;
        instance.position = {float(p.x), float(p.y), float(p.z)};
        instance.orientation = {float(std::cos(p.yaw * .5)), 0.f, 0.f, float(std::sin(p.yaw * .5))};
        instance.scale = glm::vec3(float(p.scale));
        instance.colorWeight = tint ? float(std::fmod(p.x * 0.37 + p.y * 0.71, 1.0) * 0.5 + 0.25) : 0.f;
        instances.push_back(instance);
      }
    if (instances.empty()) return;
    auto visual = viewer.addInstancedVisuals(name, raisim::Shape::Mesh, glm::vec3(1), glm::vec4(1), tint ? glm::vec4(.85f, .92f, .75f, 1) : glm::vec4(1), mesh.string(), true);
    visual->addInstances(instances);
    visual->setAutomaticMeshLodEnabled(true);
    if (foliage) {
      visual->setProjectedLodPolicy(true, 2, 8);
      visual->setShadowFoliageLodPolicy(true, 2.5f, 16);
      visual->setSortTransparentInstances(false);
      visual->configureFoliageWind(0, sway, sway > 1 ? .025f : .09f, .7f, .2f);
      visual->setFoliageImpostorPolicy(false);
    }
    if (tile >= 0) visual->setFoliageDistanceFade(55.f, 75.f, false);
    keep.push_back(visual);
  };
  // Batches in 64 m tiles: rayrai picks one mesh LOD per batch from its nearest instance, and culls
  // whole batches out of view, so a batch spread over the circuit would draw every instance at full
  // detail.
  auto tiled = [&](const std::string& name, const std::filesystem::path& mesh, const std::vector<offroad::Placement>& all, int type, bool tint) {
    for (int tile = 0; tile < 16; ++tile) instanced(name + " " + std::to_string(tile), mesh, all, type, false, 0, tile, 64.0, 4, tint);
  };
  for (int t = 0; t < 3; ++t) instanced(offroad::kTreeNames[t], forest / offroad::kTreeNames[t] / "model.gltf", scenery.trees, t, true, 1.3f);
  for (int t = 0; t < offroad::kRockTypes; ++t) tiled("rock " + std::to_string(t), offroad::rockFolder(rsc, t) / "model.gltf", scenery.rocks, t, true);
  for (int t = 0; t < int(offroad::kStoneNames.size()); ++t) tiled(offroad::kStoneNames[t], props / offroad::kStoneNames[t] / "model.gltf", scenery.stones, t, true);
  tiled("fallen trunk", props / "dead_tree_trunk_02" / "model.gltf", scenery.logs, 0, true);
  // The venue: textured props instanced, plain-coloured ones as single visuals.
  int count = 0;
  for (const auto& set : venue.props) {
    const std::string name = "venue " + std::to_string(count++) + " " + set.mesh.parent_path().filename().string();
    if (set.instanced) {
      tiled(name, set.mesh, set.at, -1, false);
      continue;
    }
    for (size_t i = 0; i < set.at.size(); ++i) {
      const auto& p = set.at[i];
      auto visual = viewer.addVisualMesh(name + " " + std::to_string(i), set.mesh.string());
      visual->setPosition(p.x, p.y, p.z);
      visual->setOrientation(glm::vec4(float(std::cos(p.yaw * .5)), 0.f, 0.f, float(std::sin(p.yaw * .5))));
      visual->setAutomaticMeshLodEnabled(true);
      singles.push_back(visual);
    }
  }
  // Grass in 32 m tiles, one batch per tile and plant: whole tiles beyond the fade distance are
  // skipped, and every batch stays small enough for rayrai's GPU instance culling.
  constexpr int kTiles = 8;
  constexpr double kTileSize = 32.0;
  grass.resize(kTiles * kTiles);
  for (int tile = 0; tile < kTiles * kTiles; ++tile) {
    grass[tile].centre = glm::vec2(float((tile % kTiles + 0.5 - kTiles / 2.0) * kTileSize), float((tile / kTiles + 0.5 - kTiles / 2.0) * kTileSize));
    for (int t = 0; t < int(offroad::kGrassNames.size()); ++t) {
      const size_t before = keep.size();
      instanced(std::string(offroad::kGrassNames[t]) + " " + std::to_string(tile), forest / offroad::kGrassNames[t] / "model.gltf", scenery.grass, t, true, .5f, tile, kTileSize, kTiles);
      if (keep.size() > before) grass[tile].batches.push_back(keep.back());
    }
  }
}

/// Keyboard state of the arrow keys (and WASD), with key-like ramps for throttle and steering.
struct KeyboardDriver {
  double throttle = 0, brake = 0, steer = 0;
  bool reverse = false;

  offroad::DriverInput update(double dt, double speed) {
    const Uint8* keys = SDL_GetKeyboardState(nullptr);
    const bool up = keys[SDL_SCANCODE_UP] || keys[SDL_SCANCODE_W];
    const bool down = keys[SDL_SCANCODE_DOWN] || keys[SDL_SCANCODE_S];
    const bool left = keys[SDL_SCANCODE_LEFT] || keys[SDL_SCANCODE_A];
    const bool right = keys[SDL_SCANCODE_RIGHT] || keys[SDL_SCANCODE_D];
    // Down brakes while rolling forward and drives backwards once stopped; up does the reverse.
    if (down && speed < 0.8) reverse = true;
    if (up && speed > -0.8) reverse = false;
    const bool accelerate = reverse ? down : up;
    const bool braking = reverse ? (up && speed < -0.3) : (down && speed > 0.3);
    auto ramp = [dt](double value, double target, double rise, double fall) {
      const double rate = target > value ? rise : fall;
      return value + std::clamp(target - value, -rate * dt, rate * dt);
    };
    throttle = ramp(throttle, accelerate ? 1.0 : 0.0, 2.5, 6.0);
    brake = ramp(brake, braking ? 1.0 : 0.0, 4.0, 8.0);
    const double target = left == right ? 0.0 : left ? 1.0 : -1.0;
    steer = ramp(steer, target, 2.2, 3.5);
    offroad::DriverInput input;
    input.throttle = throttle;
    input.brake = brake;
    input.steer = steer;
    input.reverse = reverse;
    input.handbrake = keys[SDL_SCANCODE_SPACE];
    return input;
  }
};

void drawHud(const offroad::Buggy& buggy, const offroad::TerrainInfo& info, const offroad::DriverInput& input, CameraMode cameraMode, double stepMicros, double realTimeFactor, bool showHelp) {
  const auto& t = buggy.telemetry();
  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  const ImVec2 origin = viewport->Pos, size = viewport->Size;
  const ImGuiWindowFlags panel =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.f);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.03f, 0.04f, 0.05f, 0.72f));

  // Title and controls.
  ImGui::SetNextWindowPos(ImVec2(origin.x + 12, origin.y + 12));
  ImGui::Begin("##title", nullptr, panel);
  ImGui::TextColored(ImVec4(1.f, .62f, .15f, 1.f), "RaiSim off-road buggy");
  ImGui::TextDisabled("double wishbones, coilovers, rack and pinion: URDF + pin/equality/mimic constraints");
  if (showHelp) {
    ImGui::Separator();
    ImGui::Text("Arrow keys / WASD  drive, brake, reverse, steer");
    ImGui::Text("Space  handbrake     R  reset to the track");
    ImGui::Text("C  camera (%s)     H  hide help     P  pause", cameraModeName(cameraMode));
    ImGui::Text("Mouse  drag to orbit, wheel to zoom");
  }
  ImGui::End();

  // Speed and driver inputs.
  ImGui::SetNextWindowPos(ImVec2(origin.x + 12, origin.y + size.y - 12), 0, ImVec2(0, 1));
  ImGui::Begin("##speed", nullptr, panel);
  ImGui::SetWindowFontScale(2.2f);
  ImGui::Text("%3.0f", std::abs(t.speed) * 3.6);
  ImGui::SetWindowFontScale(1.f);
  ImGui::SameLine();
  ImGui::BeginGroup();
  ImGui::Text("km/h %s", input.reverse ? "R" : "D");
  ImGui::Text("%5.1f kW", t.motorPower / 1000);
  ImGui::EndGroup();
  ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(.2f, .85f, .35f, 1.f));
  ImGui::ProgressBar(float(input.throttle), ImVec2(190, 8), "");
  ImGui::PopStyleColor();
  ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(.95f, .2f, .15f, 1.f));
  ImGui::ProgressBar(float(input.brake), ImVec2(190, 8), "");
  ImGui::PopStyleColor();
  ImGui::Text("steering wheel %+5.0f deg  rack %+5.1f mm", t.steeringWheelAngle * 180 / M_PI, t.rackPosition * 1000);
  ImGui::Text("roll %+4.1f deg  pitch %+4.1f deg", t.rollAngle * 180 / M_PI, t.pitchAngle * 180 / M_PI);
  ImGui::End();

  // Suspension: a plan view of the car with the four corners.
  ImGui::SetNextWindowPos(ImVec2(origin.x + size.x - 12, origin.y + 12), 0, ImVec2(1, 0));
  ImGui::Begin("##suspension", nullptr, panel);
  ImGui::TextColored(ImVec4(1.f, .62f, .15f, 1.f), "Suspension");
  // Columns: text | travel bar | wheel | body | wheel | travel bar | text.
  const float text = 128, ch = 150, barW = 12, wheelX = text + barW + 26, bodyX = wheelX + 46;
  const float width = 2 * bodyX;
  ImDrawList* draw = ImGui::GetWindowDrawList();
  const ImVec2 base = ImGui::GetCursorScreenPos();
  ImGui::Dummy(ImVec2(width, 2 * ch));
  const ImVec2 body0(base.x + bodyX - 22, base.y + 28), body1(base.x + bodyX + 22, base.y + 2 * ch - 28);
  draw->AddRectFilled(body0, body1, IM_COL32(60, 64, 70, 255), 8.f);
  draw->AddTriangleFilled(ImVec2(base.x + bodyX, base.y + 12), ImVec2(base.x + bodyX - 10, base.y + 26), ImVec2(base.x + bodyX + 10, base.y + 26), IM_COL32(255, 160, 40, 255));
  for (int i = 0; i < 4; ++i) {
    const auto& c = t.corners[i];
    const bool right = i % 2, rear = i >= 2;
    const float top = base.y + (rear ? ch : 0);
    auto column = [&](float x) { return base.x + (right ? width - x : x); };
    // Wheel, turned by its steer angle; red while it is off the ground.
    const ImVec2 wc(column(wheelX), top + ch / 2);
    const float a = float(-c.steerAngle);
    auto rot = [&](float x, float y) { return ImVec2(wc.x + x * std::cos(a) - y * std::sin(a), wc.y + x * std::sin(a) + y * std::cos(a)); };
    draw->AddQuadFilled(rot(-6, -18), rot(6, -18), rot(6, 18), rot(-6, 18), c.contact ? IM_COL32(20, 20, 22, 255) : IM_COL32(150, 40, 40, 255));
    draw->AddLine(ImVec2(wc.x + (right ? -6.f : 6.f), wc.y), ImVec2(column(bodyX - 22), wc.y), IM_COL32(150, 150, 150, 255), 2.f);
    // Damper compression bar, in the spring's colour.
    const glm::vec3 colour = springColour(c.compression);
    const float barX = column(text + 6 + (right ? barW : 0));
    const ImVec2 bar0(barX, top + 14), bar1(barX + barW, top + ch - 14);
    draw->AddRect(ImVec2(bar0.x - 1, bar0.y - 1), ImVec2(bar1.x + 1, bar1.y + 1), IM_COL32(180, 180, 180, 160));
    draw->AddRectFilled(ImVec2(bar0.x, bar1.y - (bar1.y - bar0.y) * float(c.compression)), bar1, imColour(colour));
    ImGui::SetCursorScreenPos(ImVec2(right ? column(text) : base.x, top + 8));
    ImGui::BeginGroup();
    ImGui::TextColored(ImVec4(colour.r, colour.g, colour.b, 1), "%s", kCornerLabels[i]);
    ImGui::Text("travel %+5.0f mm", -c.shockExtension * 1000);
    ImGui::Text("spring %5.2f kN", c.springForce / 1000);
    ImGui::Text("damper %+5.2f kN", c.damperForce / 1000);
    ImGui::Text("load   %5.2f kN", c.wheelLoad / 1000);
    ImGui::Text("tyre   %4.1f mm", c.tyreDeflection * 1000);
    ImGui::Text("camber %+4.1f deg", c.camber * 180 / M_PI);
    if (i < 2)
      ImGui::Text("steer  %+4.1f deg", c.steerAngle * 180 / M_PI);
    else
      ImGui::Text("slip   %+4.0f %%", c.slip * 100);
    ImGui::EndGroup();
  }
  ImGui::SetCursorScreenPos(ImVec2(base.x, base.y + 2 * ch + 6));
  ImGui::TextDisabled("spring colour: blue full droop, green ride height, red bump stop");
  ImGui::End();

  // Physics: Newton-Euler balances of the measured wheel loads, live.
  ImGui::SetNextWindowPos(ImVec2(origin.x + size.x - 12, origin.y + size.y - 12), 0, ImVec2(1, 1));
  ImGui::Begin("##physics", nullptr, panel);
  ImGui::TextColored(ImVec4(1.f, .62f, .15f, 1.f), "Physics");
  double loads = 0, roll = 0, pitch = 0, spin = 0;
  for (const auto& c : t.corners) {
    loads += c.wheelLoad;
    // Moments of the vertical loads about the centre of mass (heading frame).
    roll += c.wheelLoad * (c.contact2d.y() - t.com2d.y());
    pitch += c.wheelLoad * (c.contact2d.x() - t.com2d.x());
    spin += c.wheelSpeed;
  }
  const double m = buggy.mass(), yawRate = buggy.system()->getGeneralizedVelocity()[5];
  ImGui::Text("wheel loads %5.2f kN = %5.3f x weight", loads / 1000, loads / (m * 9.81 * std::cos(t.rollAngle) * std::cos(t.pitchAngle)));
  ImGui::TextDisabled("load transfer (equal in steady state): moments of the wheel loads");
  // Steady state: the tyre forces at the ground, h below the centre of mass, balance these.
  ImGui::Text("  roll  %+6.2f kNm   m a_y h + I w r  %+6.2f kNm", -roll / 1000, (m * t.acceleration.y() * t.comHeight + 1.25 * spin * yawRate) / 1000);
  ImGui::Text("  pitch %+6.2f kNm   -m a_x h         %+6.2f kNm", pitch / 1000, -m * t.acceleration.x() * t.comHeight / 1000);
  const auto& fl = t.corners[0];
  const auto& fr = t.corners[1];
  if (std::abs(fl.steerAngle) > 0.03) {
    // Ackermann: cot(outer) - cot(inner) = kingpin track / wheelbase.
    const double inner = std::max(std::abs(fl.steerAngle), std::abs(fr.steerAngle));
    const double outer = std::min(std::abs(fl.steerAngle), std::abs(fr.steerAngle));
    const double ideal = std::atan(1.0 / (1.0 / std::tan(inner) + 2.0258 / 2.7142));
    ImGui::Text("Ackermann: inner %4.1f, outer %4.1f (ideal %4.1f) deg", inner * 180 / M_PI, outer * 180 / M_PI, ideal * 180 / M_PI);
  } else {
    ImGui::TextDisabled("Ackermann: steer to compare the front wheel angles");
  }
  ImGui::Text("loop closure error %6.2f um (8 pins, 4 equalities, 1 mimic)", t.maxLoopError * 1e6);
  ImGui::Text("physics %5.1f us per 1 ms step, %4.1fx real time", stepMicros, realTimeFactor);
  ImGui::End();

  // Minimap.
  ImGui::SetNextWindowPos(ImVec2(origin.x + size.x * 0.5f, origin.y + size.y - 12), 0, ImVec2(0.5f, 1));
  ImGui::Begin("##map", nullptr, panel);
  const float map = 150;
  const ImVec2 p0 = ImGui::GetCursorScreenPos();
  ImGui::Dummy(ImVec2(map, map));
  auto toMap = [&](double x, double y) { return ImVec2(p0.x + float((x / info.extent + 0.5) * map), p0.y + float((0.5 - y / info.extent) * map)); };
  draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(p0, ImVec2(p0.x + map, p0.y + map), IM_COL32(48, 70, 40, 200), 6.f);
  for (const auto* area : {&info.pit, &info.spectators}) draw->AddRectFilled(toMap(area->x0, area->y1), toMap(area->x1, area->y0), IM_COL32(150, 150, 140, 200), 2.f);
  std::vector<ImVec2> line;
  for (const auto& p : info.centre) line.push_back(toMap(p.x(), p.y()));
  draw->AddPolyline(line.data(), int(line.size()), IM_COL32(190, 150, 100, 255), ImDrawFlags_Closed, 3.f);
  const auto& q = buggy.system()->getGeneralizedCoordinate();
  const double heading = std::atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] * q[5] + q[6] * q[6]));
  const ImVec2 car = toMap(q[0], q[1]);
  const float c = float(std::cos(heading)), s = float(std::sin(heading));
  draw->AddTriangleFilled(ImVec2(car.x + 7 * c, car.y - 7 * s), ImVec2(car.x - 5 * c + 4 * s, car.y + 5 * s + 4 * c), ImVec2(car.x - 5 * c - 4 * s, car.y + 5 * s - 4 * c),
                          IM_COL32(255, 160, 40, 255));
  ImGui::End();
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();
}

int runViewer(const Options& options, char* argv0) {
  const std::filesystem::path root = exampleRscPath(argv0, "offroad_buggy");
  const std::filesystem::path rsc = root.parent_path();
  const auto info = offroad::readTerrainInfo(root / "terrain" / "terrain.txt");
  const auto surface = offroad::readSurface(root / "terrain", info.extent);

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(kTimeStep);
  offroad::setMaterials(*world);
  const auto heights = offroad::readHeights(root / "terrain", info);
  auto* terrain = offroad::addTerrain(*world, info, heights);
  offroad::PlacementGrid occupied(4.0);
  auto venue = offroad::buildVenue(*world, *terrain, info, rsc, occupied);
  const auto scenery = offroad::placeScenery(*world, *terrain, info, surface, rsc, options.grassDensity, occupied);
  offroad::Buggy buggy(*world, (root / "buggy.urdf").string());
  const offroad::TrackLocator track(info);
  auto ground = [&](double x, double y) { return terrain->getHeight(x, y); };
  buggy.place(info.start.x(), info.start.y(), info.start.z(), ground);
  std::cout << "Off-road buggy: " << buggy.system()->getDOF() << " degrees of freedom, " << scenery.trees.size() << " trees, " << scenery.rocks.size() << " boulders, " << scenery.stones.size()
            << " stones, " << scenery.grass.size() << " grass tufts\nVenue: " << venue.tyreStacks << " tyre stacks, " << venue.bales << " hay bales, " << venue.cones << " cones and " << venue.markers
            << " marker posts (dynamic), " << venue.pitBoxes << " pit boxes; " << scenery.colliders + venue.colliders << " static colliders\n";

  ExampleApp app;
  const bool capture = !options.screenshot.empty();
  if (!app.init("RaiSim off-road buggy", 1600, 900, !capture)) return 1;
  if (capture) SDL_GL_SetSwapInterval(0);
  ImGui::GetIO().IniFilename = nullptr;
  int status = 0;
  {
    Viewer viewer(world, 1600, 900);
    viewer.setAsyncMeshLoadingEnabled(true);
    viewer.setKeyboardInputEnabled(false);  // the arrow keys and WASD drive the car
    configureRendering(viewer, rsc / "city" / "sky" / "kloofendal_48d_partly_cloudy_puresky_4k.hdr");
    // The terrain is drawn by the visuals below; the height map stays the collision surface.
    viewer.setObjectHidden(terrain, true);
    const auto terrainVisuals = addTerrainVisuals(viewer, info, heights, offroad::terrainColours(*terrain, info, surface), surface, root / "terrain" / "textures");
    std::vector<std::shared_ptr<raisin::InstancedVisuals>> instanced;
    std::vector<std::shared_ptr<raisin::Visuals>> singles;
    std::vector<GrassTile> grassTiles;
    addScenery(viewer, scenery, venue, rsc, instanced, singles, grassTiles);
    // Hay bales and cones: their visual meshes follow the bodies.
    std::vector<std::shared_ptr<raisin::Visuals>> looseVisuals;
    for (size_t i = 0; i < venue.loose.size(); ++i) looseVisuals.push_back(viewer.addVisualMesh("loose " + std::to_string(i), venue.loose[i].mesh.string()));
    for (auto& visual : looseVisuals) visual->setAutomaticMeshLodEnabled(true);
    offroad::SuspensionVisuals suspension(*buggy.system());
    auto& camera = viewer.getCamera();
    camera.zoom = 50;
    camera.nearPlane = 0.05f;
    camera.target = {float(info.start.x()), float(info.start.y()), float(terrain->getHeight(info.start.x(), info.start.y()) + 0.8)};
    camera.yaw = float(info.start.z() * 180 / M_PI);
    camera.pitch = -13;
    camera.position = camera.target - 7.5f * glm::vec3(std::cos(info.start.z()), std::sin(info.start.z()), -0.3f);
    camera.setCameraFixedTarget(true);
    camera.setCameraFixedDistance(true);

    RayraiLoadingProgress loading("Loading the race track");
    loading.update(viewer.pendingAsyncMeshLoadCount());
    KeyboardDriver keyboard;
    offroad::Autopilot pilot(info, track);
    CameraMode cameraMode = CameraMode(std::clamp(options.cameraMode, 0, 3)), lastCameraMode = CameraMode::Chase;
    std::array<float, 4> distances;
    for (int m = 0; m < 4; ++m) distances[m] = kCameraViews[m].distance;
    bool paused = false, showHelp = true;
    double lastHeading = info.start.z(), stepMicros = 0, realTimeFactor = 0;
    glm::vec3 smoothTarget = camera.target;
    int captureFrames = 0;
    auto captureStart = std::chrono::steady_clock::now();
    auto last = std::chrono::steady_clock::now();
    offroad::DriverInput input;
    while (!app.quit) {
      app.processEvents();
      if (app.quit) break;
      const auto now = std::chrono::steady_clock::now();
      const double frameTime = capture ? 1.0 / 60 : std::min(std::chrono::duration<double>(now - last).count(), 1.0 / 20);
      last = now;
      if (!ImGui::GetIO().WantTextInput) {
        if (ImGui::IsKeyPressed(ImGuiKey_C, false)) cameraMode = CameraMode((int(cameraMode) + 1) % 4);
        if (ImGui::IsKeyPressed(ImGuiKey_H, false)) showHelp = !showHelp;
        if (ImGui::IsKeyPressed(ImGuiKey_P, false)) paused = !paused;
        if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
          // Back onto the closest point of the track, facing along it.
          const auto& q = buggy.system()->getGeneralizedCoordinate();
          const auto [distance, station] = track.locate(q[0], q[1]);
          Eigen::Vector3d p;
          Eigen::Vector2d d;
          track.at(distance < 40 ? station : 40.0, p, d);
          buggy.place(p.x(), p.y(), std::atan2(d.y(), d.x()), ground);
        }
      }
      const bool ready = !loading.visible();
      if (ready && !paused) {
        input = capture ? pilot.update(buggy) : keyboard.update(frameTime, buggy.telemetry().speed);
        const int steps = std::clamp(int(std::round(frameTime / kTimeStep)), 1, 50);
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < steps; ++i) {
          buggy.applyControls(input, kTimeStep);
          world->integrate();
          buggy.accumulate(kTimeStep);
        }
        const double spent = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        stepMicros += (spent / steps * 1e6 - stepMicros) * 0.05;
        realTimeFactor = kTimeStep * 1e6 / std::max(stepMicros, 1e-3);
        buggy.updateTelemetry(steps * kTimeStep);
        const auto& q = buggy.system()->getGeneralizedCoordinate();
        offroad::updateLooseProps(venue, Eigen::Vector3d(q[0], q[1], q[2]), steps * kTimeStep);
      }
      auto* car = buggy.system();
      suspension.update();
      for (size_t i = 0; i < venue.loose.size(); ++i) {
        const auto& prop = venue.loose[i];
        const Eigen::Matrix3d R = prop.body->getRotationMatrix();
        const Eigen::Vector3d base = prop.body->getPosition() - R.col(2) * prop.lift;
        looseVisuals[i]->setPosition(base.x(), base.y(), base.z());
        looseVisuals[i]->setOrientation(toGlmQuat(R));
      }
      // Camera: follows the car, turns with it, and eases back to the mode's view unless the user
      // is orbiting with the mouse.
      const auto& q = car->getGeneralizedCoordinate();
      const double heading = std::atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] * q[5] + q[6] * q[6]));
      const bool dragging = SDL_GetMouseState(nullptr, nullptr) & (SDL_BUTTON_LMASK | SDL_BUTTON_RMASK);
      if (cameraMode != CameraMode::Free) {
        const auto& view = kCameraViews[int(cameraMode)];
        raisim::Mat<3, 3> R;
        raisim::Vec<3> p;
        car->getBodyPose(0, R, p);
        const Eigen::Vector3d focus = p.e() + R.e() * view.focus;
        const glm::vec3 f(focus.x(), focus.y(), focus.z());
        smoothTarget = glm::mix(smoothTarget, f, float(std::min(1.0, frameTime * 12)));
        camera.target = smoothTarget;
        camera.yaw += float(std::remainder(heading - lastHeading, 2 * M_PI) * 180 / M_PI);
        if (!dragging) {
          const double error = std::remainder(heading * 180 / M_PI + view.yaw - camera.yaw, 360.0);
          camera.yaw += float(error * std::min(1.0, frameTime * 1.5));
          camera.pitch += float((view.pitch - camera.pitch) * std::min(1.0, frameTime * 1.5));
        }
        if (cameraMode != lastCameraMode) {
          distances[int(lastCameraMode)] = camera.cameraFixedDistance();
          setOrbitDistance(camera, distances[int(cameraMode)]);
        }
        // Stay above the ground: look down more steeply where the orbit would dip below it.
        for (int k = 0; k < 30; ++k) {
          const float yaw = glm::radians(camera.yaw), pitch = glm::radians(camera.pitch);
          const glm::vec3 front(std::cos(yaw) * std::cos(pitch), std::sin(yaw) * std::cos(pitch), std::sin(pitch));
          const glm::vec3 eye = camera.target - front * camera.cameraFixedDistance();
          if (eye.z > terrain->getHeight(eye.x, eye.y) + 0.35 || camera.pitch < -80) break;
          camera.pitch -= 1.f;
        }
      }
      lastCameraMode = cameraMode;
      lastHeading = heading;

      updateGrassTiles(grassTiles, camera.position);
      app.beginFrame();
      app.renderViewer(viewer);
      loading.update(viewer.pendingAsyncMeshLoadCount());
      if (!capture || options.hud) drawHud(buggy, info, input, cameraMode, stepMicros, realTimeFactor, showHelp);
      if (!capture) loading.draw();
      const bool last = capture && ready && captureFrames + 1 == options.captureFrames;
      if (last && options.hud) {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        status = saveWindowScreenshot(app.window, options.screenshot) ? 0 : 1;
        SDL_GL_SwapWindow(app.window);
        break;
      }
      app.endFrame();
      if (!options.view.empty()) {
        camera.setCameraFixedTarget(false);
        camera.setCameraFixedDistance(false);
        camera.position = {options.view[0], options.view[1], options.view[2]};
        camera.yaw = options.view[3];
        camera.pitch = options.view[4];
      }
      if (capture && ready && captureFrames == 0) captureStart = std::chrono::steady_clock::now();
      if (capture && ready && captureFrames == options.captureFrames / 2) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - captureStart).count();
        std::cout << "frame time " << ms / std::max(1, captureFrames) << " ms (physics included)\n";
      }
      if (capture && ready && ++captureFrames == options.captureFrames) {
        status = raisim_examples::saveCameraPng(viewer.getCamera(), options.screenshot) ? 0 : 1;
        break;
      }
    }
  }  // Release graphics resources before closing the OpenGL context.
  app.shutdown();
  return status;
}

}  // namespace

int main(int argc, char* argv[]) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--screenshot" && i + 1 < argc)
      options.screenshot = argv[++i];
    else if (option == "--validate")
      options.validate = true;
    else if (option == "--grass" && i + 1 < argc)
      options.grassDensity = std::stod(argv[++i]);
    else if (option == "--frames" && i + 1 < argc)
      options.captureFrames = std::stoi(argv[++i]);
    else if (option == "--camera" && i + 1 < argc)
      options.cameraMode = std::stoi(argv[++i]);
    else if (option == "--hud")
      options.hud = true;
    else if (option == "--view" && i + 5 < argc)
      for (int k = 0; k < 5; ++k) options.view.push_back(std::stof(argv[++i]));
    else {
      std::cerr << "Usage: rayrai_offroad_buggy [--screenshot PNG] [--validate] [--grass DENSITY]\n";
      return 2;
    }
  }
  try {
    if (options.validate) return offroad::validate(exampleRscPath(argv[0], "offroad_buggy"));
    return runViewer(options, argv[0]);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
