// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#pragma once

#include "levels.hpp"
#include "raisim/Rasset.hpp"
#include "rayrai/OpenGLMesh.hpp"
#include "rayrai/Visuals.hpp"

#ifndef TANK_SURVIVAL_BROADLEAF_TREES
#define TANK_SURVIVAL_BROADLEAF_TREES 1
#endif

namespace tank {

// Whether broadleaf trees grow among the pines and firs.
inline constexpr bool kBroadleafTrees = TANK_SURVIVAL_BROADLEAF_TREES != 0;

// How dense the grass is at each render quality (Fast to Ultra), how far it is drawn (m), and how fine
// the tree and shrub meshes and how small plants on screen may become before they are simplified or
// thinned out (pixels).
inline constexpr std::array<double, 4> kGrassDensity{0.12, 0.3, 0.55, 1.0};
inline constexpr std::array<float, 4> kFoliageReach{60.f, 90.f, 130.f, 170.f};
inline constexpr std::array<float, 4> kFoliagePixelError{2.f, 1.25f, 0.75f, 0.5f};
inline constexpr std::array<float, 4> kFoliageMinPixels{2.f, 1.5f, 1.f, 0.75f};

// The sun, sky, haze and weather of a level, lit for photographs: ACES tone mapping, the generated sky
// as both the background and the image-based light, and a sun casting cascaded shadows.
inline void configureRendering(Viewer& viewer, const LevelSpec& level, int quality) {
  auto settings = Viewer::defaultRenderQualitySettings(raisin::RenderQualityPreset(quality));
  settings.gamma = 2.2f;
  settings.colorMode = raisin::ViewerColorMode::AcesApprox;
  settings.pbrExposure = 1.05f;
  settings.pbrEnvironmentLightingTint = glm::vec3(0.65f);
  settings.depthOfFieldEnabled = false;
  settings.reflectiveGround = false;
  settings.addViewerFillLights = false;
  settings.directionalShadowCascadeCount = quality >= 2 ? 3 : 2;
  settings.directionalShadowCascadeMaxDistance = quality >= 2 ? 120.0f : 70.0f;
  if (quality >= 2) settings.highFidelityPbr = true;
  viewer.setRenderQualitySettings(settings);
  Viewer::WeatherSettings weather;
  weather.enabled = true;
  weather.preset = Viewer::WeatherPreset::Custom;
  weather.useExplicitSunAngles = true;
  weather.sunAzimuthDegrees = level.sunAzimuth;
  weather.sunElevationDegrees = level.sunElevation;
  weather.sunDiskSize = 0.004f;
  weather.windSpeed = float(level.wind.norm());
  weather.cloudCoverage = level.clouds;
  weather.cloudDensity = 0.35f + 0.5f * level.clouds;
  weather.visibilityMeters = level.visibility;
  weather.airTurbidity = level.turbidity;
  weather.snowCoverage = level.snow;
  weather.precipitationRate = level.rain;
  weather.wetness = level.rain > 0.0f ? 0.6f : 0.0f;
  weather.lightningRate = level.lightning;
  weather.fogColor = level.fogColor;
  viewer.setWeatherSettings(weather);
  // The sky behind the scene at its own brightness; the weather would dim it twice for a low sun. Over
  // snow, which reflects most of the light, the sky is brightened to match, as the eye adapts.
  const auto sky = viewer.generateWeatherSkyEnvironment(256, 32, false);
  viewer.setEnvironmentBackground(sky.environmentMap, 1.0f + 0.9f * level.snow);
  const float azimuth = glm::radians(level.sunAzimuth), elevation = glm::radians(level.sunElevation);
  auto& sun = viewer.getLight();
  sun.type = raisin::LightType::DIRECTIONAL;
  sun.setShadowsEnabled(true);
  sun.setShadowResolution(quality >= 2 ? 2048 : 1024);
  sun.setShadowParams(0.0008f, 1.f, 1.25f);
  viewer.setShadowOrtho(45, 0.1f, 260);
  sun.direction = -glm::vec3(std::cos(elevation) * std::cos(azimuth), std::cos(elevation) * std::sin(azimuth), std::sin(elevation));
  // Low suns are redder and weaker, as through more air.
  const float low = std::clamp(1.0f - level.sunElevation / 35.0f, 0.0f, 1.0f);
  const glm::vec3 tint = glm::mix(glm::vec3(1.0f, 0.95f, 0.86f), glm::vec3(1.0f, 0.62f, 0.38f), low);
  sun.diffuse = tint * (17.0f - 7.0f * low) * (1.0f - 0.45f * level.clouds);
  sun.specular = sun.diffuse;
  sun.ambient = glm::vec3(0.13f);
  // Explosions throw up the level's own ground: soil, sand, concrete dust or snow, under black soot.
  auto effects = viewer.getExplosionEffectSettings();
  effects.smokeColor = glm::vec3(0.07f, 0.066f, 0.062f);
  switch (level.terrain) {
    case Terrain::Dunes: effects.dustColor = {0.33f, 0.25f, 0.19f}; break;
    case Terrain::Town: effects.dustColor = {0.2f, 0.17f, 0.14f}; break;
    case Terrain::Ridges: effects.dustColor = {0.32f, 0.31f, 0.31f}; break;
    case Terrain::Fortress: effects.dustColor = {0.22f, 0.15f, 0.085f}; break;
    case Terrain::Hills:
    default: effects.dustColor = {0.19f, 0.13f, 0.075f}; break;
  }
  viewer.setExplosionEffectSettings(effects);
}

// A ground layer: a Poly Haven texture set, how many metres one repeat covers, a tint and its finish.
struct GroundLayer {
  std::string albedo, normal;
  double tile;
  glm::vec4 color;
  float roughness, normalStrength;
};

// The three layers of each terrain: the base, patches cut in by a noise field, and rock on steep slopes.
inline std::array<GroundLayer, 3> groundLayers(Terrain terrain, const std::string& assets) {
  const auto offroad = [&](const std::string& name) {
    return std::array<std::string, 2>{assets + "/offroad_buggy/terrain/textures/" + name + "_diff_2k.jpg",
                                      assets + "/offroad_buggy/terrain/textures/" + name + "_nor_gl_1k.jpg"};
  };
  const auto own = [&](const std::string& name) {
    return std::array<std::string, 2>{assets + "/tank_survival/ground/" + name + "_diffuse_1k.jpg",
                                      assets + "/tank_survival/ground/" + name + "_nor_gl_1k.jpg"};
  };
  const auto layer = [](const std::array<std::string, 2>& files, double tile, glm::vec4 color, float roughness,
                        float normal = 1.0f) { return GroundLayer{files[0], files[1], tile, color, roughness, normal}; };
  const auto rock = layer(offroad("rock_face_03"), 6.0, {0.9f, 0.88f, 0.85f, 1.f}, 0.85f, 1.2f);
  switch (terrain) {
    case Terrain::Dunes:
      return {layer(own("aerial_sand"), 9.0, {0.8f, 0.66f, 0.5f, 1.f}, 0.95f, 0.9f),
              layer(offroad("dry_ground_rocks"), 4.5, {0.8f, 0.68f, 0.54f, 1.f}, 0.92f), rock};
    case Terrain::Town:
      return {layer(own("cracked_concrete"), 5.0, {0.9f, 0.9f, 0.9f, 1.f}, 0.9f),
              layer(own("rubble"), 3.5, {0.95f, 0.93f, 0.9f, 1.f}, 0.9f), rock};
    case Terrain::Ridges:
      return {layer(own("snow_02"), 6.0, {1.0f, 1.0f, 1.0f, 1.f}, 0.7f, 0.7f),
              layer(offroad("rocky_trail"), 3.0, {0.5f, 0.5f, 0.52f, 1.f}, 0.9f), rock};
    case Terrain::Fortress:
      return {layer(offroad("rocky_terrain_02"), 7.0, {1.f, 1.f, 1.f, 1.f}, 0.95f, 0.8f),
              layer(own("burned_ground_01"), 4.0, {1.f, 1.f, 1.f, 1.f}, 0.95f), rock};
    case Terrain::Hills:
    default:
      return {layer(offroad("rocky_terrain_02"), 7.0, {1.f, 1.f, 1.f, 1.f}, 0.95f, 0.8f),
              layer(offroad("brown_mud_dry"), 3.0, {0.92f, 0.86f, 0.8f, 1.f}, 0.96f), rock};
  }
}

// The terrain drawn as textured meshes over the heightmap (which is hidden), in chunks that are built
// again when explosions dig craters. Each grid triangle is cut along the 0.5 contours of the rock mask
// (from the slope) and the patch mask (from a noise field), and each piece is drawn with its layer.
class TerrainVisuals {
 public:
  void build(Viewer& viewer, const raisim::HeightMap& terrain, Terrain kind, const std::string& assets) {
    clear(viewer);
    kind_ = kind;
    samples_ = int(terrain.getXSamples());
    extent_ = terrain.getXSize();
    const auto layers = groundLayers(kind, assets);
    for (int l = 0; l < 3; ++l) {
      auto& m = materials_[l] = raisin::Material::pbr("ground " + std::to_string(l), layers[l].color, 0.f, layers[l].roughness);
      m.albedoMap = Viewer::loadTextureWithTiling(layers[l].albedo.c_str(), true);
      m.normalMap = Viewer::loadTextureWithTiling(layers[l].normal.c_str(), false);
      m.normalStrength = layers[l].normalStrength;
      m.useVertexColorAsAlbedo = true;
      tiles_[l] = layers[l].tile;
    }
    chunkVisuals_.assign(size_t(kChunks * kChunks), {});
    for (int c = 0; c < kChunks * kChunks; ++c) buildChunk(viewer, terrain, c);
    buildOuter(viewer);
  }

  // Marks the chunks a crater of the given radius at (x, y) reaches; flush() builds them again.
  void dig(double x, double y, double radius) {
    const double cell = extent_ / (samples_ - 1), half = 0.5 * extent_;
    const int per = (samples_ - 1) / kChunks;
    const int i0 = std::clamp(int((x - radius + half) / cell) - 1, 0, samples_ - 2) / per;
    const int i1 = std::clamp(int((x + radius + half) / cell) + 1, 0, samples_ - 2) / per;
    const int j0 = std::clamp(int((y - radius + half) / cell) - 1, 0, samples_ - 2) / per;
    const int j1 = std::clamp(int((y + radius + half) / cell) + 1, 0, samples_ - 2) / per;
    for (int cy = j0; cy <= std::min(j1, kChunks - 1); ++cy)
      for (int cx = i0; cx <= std::min(i1, kChunks - 1); ++cx) dirty_.push_back(cy * kChunks + cx);
  }

  void flush(Viewer& viewer, const raisim::HeightMap& terrain) {
    std::sort(dirty_.begin(), dirty_.end());
    dirty_.erase(std::unique(dirty_.begin(), dirty_.end()), dirty_.end());
    for (int chunk : dirty_) buildChunk(viewer, terrain, chunk);
    dirty_.clear();
  }

  void clear(Viewer& viewer) {
    for (auto& chunk : chunkVisuals_)
      for (const auto& name : chunk) viewer.removeVisualObject(name);
    chunkVisuals_.clear();
    dirty_.clear();
    for (const auto& name : outer_) viewer.removeVisualObject(name);
    outer_.clear();
  }

  // The fraction of rock (steep ground) at (x, y), for placing plants and stones.
  double rock(const raisim::HeightMap& terrain, double x, double y) const { return rockMask(terrain, x, y); }

 private:
  static constexpr int kChunks = 10;

  struct Point {
    double i, j;
    std::array<double, 2> mask;
  };
  struct Polygon {
    std::array<Point, 8> p;
    int size = 0;
  };

  static double smoothstep(double a, double b, double x) {
    const double t = std::clamp((x - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
  }

  double rockMask(const raisim::HeightMap& terrain, double x, double y) const {
    const double d = 1.0;
    const double dx = (terrain.getHeight(x + d, y) - terrain.getHeight(x - d, y)) / (2 * d);
    const double dy = (terrain.getHeight(x, y + d) - terrain.getHeight(x, y - d)) / (2 * d);
    const double slope = std::sqrt(dx * dx + dy * dy);
    return kind_ == Terrain::Town ? 0.0 : smoothstep(0.42, 0.62, slope);
  }

  // The color of the base ground: lush and dry patches of meadow that break up the tiling, drifts of
  // lighter and darker sand, or bluish snow.
  glm::vec3 groundTint(double x, double y) const {
    const auto wave = [](double x, double y, double k, double phase) { return std::sin(x * k + phase) * std::cos(y * k * 0.83 - phase * 1.7); };
    const double patch = 0.5 * wave(x, y, 0.045, 0.3) + 0.3 * wave(x + 13, y - 7, 0.11, 1.9) + 0.2 * wave(x - 5, y + 21, 0.27, 4.1);
    const double dry = std::clamp(0.5 + 0.6 * patch, 0.0, 1.0);
    switch (kind_) {
      case Terrain::Hills: return {float(0.72 + 0.34 * dry), float(0.86 + 0.14 * dry), float(0.66 + 0.10 * dry)};
      case Terrain::Fortress: return {float(0.95 + 0.25 * dry), float(0.85 + 0.12 * dry), float(0.66 + 0.06 * dry)};
      case Terrain::Dunes: return {float(0.92 + 0.12 * dry), float(0.9 + 0.1 * dry), float(0.88 + 0.08 * dry)};
      case Terrain::Ridges: return {float(0.9 + 0.1 * dry), float(0.93 + 0.07 * dry), 1.0f};
      case Terrain::Town:
      default: return glm::vec3(float(0.86 + 0.14 * dry));
    }
  }

  double patchMask(double x, double y) const {
    const double n = 0.5 + 0.28 * std::sin(0.11 * x + 1.3) * std::cos(0.09 * y - 0.4) + 0.18 * std::sin(0.31 * x - 0.23 * y + 2.0) +
                     0.1 * std::sin(0.73 * x + 0.61 * y);
    const double threshold = kind_ == Terrain::Fortress ? 0.45 : kind_ == Terrain::Town ? 0.5 : 0.62;
    // The river's bed and banks are mud or gravel, not meadow.
    const double bank = 1.0 - smoothstep(0.6, 2.2, riverDistance(kind_, x, y) - riverHalfWidth(kind_));
    return std::max(smoothstep(threshold - 0.08, threshold + 0.08, n), bank);
  }

  // The hills outside the heightmap out to kOuter metres, in four slabs drawn with the base ground.
  void buildOuter(Viewer& viewer) {
    constexpr double kOuter = 660.0, kCell = 5.5;
    const double half = 0.5 * extent_;
    const int cells = int(2.0 * kOuter / kCell);
    // Heights on a half-cell lattice, each computed once: the corners and the points half a cell
    // beside them that their normals take differences of (all exact in binary, so the same numbers).
    const int lattice = 2 * cells + 3;
    std::vector<double> heights(size_t(lattice) * size_t(lattice), std::numeric_limits<double>::quiet_NaN());
    const auto heightAt = [&](int hi, int hj) {
      double& h = heights[size_t(hj + 1) * size_t(lattice) + size_t(hi + 1)];
      if (std::isnan(h)) h = outerHeight(kind_, -kOuter + hi * 0.5 * kCell, -kOuter + hj * 0.5 * kCell);
      return h;
    };
    for (int slab = 0; slab < 4; ++slab) {
      std::vector<::Vertex> vertices;
      std::vector<unsigned int> indices;
      std::vector<int> index(size_t(cells + 1) * (cells + 1), -1);
      const auto inSlab = [&](double x, double y) {
        if (std::max(std::abs(x), std::abs(y)) < half - 1e-6) return false;
        switch (slab) {
          case 0: return y <= -half + 1e-6;
          case 1: return y >= half - 1e-6;
          case 2: return x <= -half + 1e-6 && std::abs(y) <= half + 1e-6;
          default: return x >= half - 1e-6 && std::abs(y) <= half + 1e-6;
        }
      };
      const auto corner = [&](int i, int j) {
        const size_t k = size_t(j) * (cells + 1) + i;
        if (index[k] >= 0) return unsigned(index[k]);
        const double x = -kOuter + i * kCell, y = -kOuter + j * kCell;
        const double z = heightAt(2 * i, 2 * j), d = 0.5 * kCell;
        ::Vertex v;
        v.Position = glm::vec3(float(x), float(y), float(z));
        v.Normal = glm::normalize(glm::vec3(float(-(heightAt(2 * i + 1, 2 * j) - heightAt(2 * i - 1, 2 * j)) / (2 * d)),
                                            float(-(heightAt(2 * i, 2 * j + 1) - heightAt(2 * i, 2 * j - 1)) / (2 * d)), 1.f));
        v.TexCoords = glm::vec2(float(x / tiles_[0]), float(y / tiles_[0]));
        // Rock shows through on the steep faces of the hills around the arena.
        const float steep = glm::smoothstep(0.99f, 0.9f, v.Normal.z);
        v.Color = glm::vec4(groundTint(x, y) * glm::mix(glm::vec3(1.f), glm::vec3(0.5f, 0.47f, 0.45f), 0.85f * steep), 1.f);
        index[k] = int(vertices.size());
        vertices.push_back(v);
        return unsigned(index[k]);
      };
      for (int j = 0; j < cells; ++j)
        for (int i = 0; i < cells; ++i) {
          const double x = -kOuter + (i + 0.5) * kCell, y = -kOuter + (j + 0.5) * kCell;
          if (!inSlab(x, y)) continue;
          const unsigned a = corner(i, j), b = corner(i + 1, j), c = corner(i, j + 1), e = corner(i + 1, j + 1);
          indices.insert(indices.end(), {a, b, e, a, e, c});
        }
      auto meshes = std::make_shared<std::vector<std::shared_ptr<::OpenGLMesh>>>();
      meshes->push_back(std::make_shared<::OpenGLMesh>(std::move(vertices), std::move(indices)));
      meshes->front()->material = materials_[0];
      outer_.push_back("outer ground " + std::to_string(slab) + " " + std::to_string(revision_++));
      viewer.addVisualCustomMesh(outer_.back(), meshes, glm::vec4(1.f), true);
    }
  }

  void buildChunk(Viewer& viewer, const raisim::HeightMap& terrain, int chunk) {
    for (const auto& name : chunkVisuals_[size_t(chunk)]) viewer.removeVisualObject(name);
    chunkVisuals_[size_t(chunk)].clear();
    const int n = samples_, per = (n - 1) / kChunks, cx = chunk % kChunks, cy = chunk / kChunks;
    const double cell = extent_ / (n - 1), half = 0.5 * extent_;
    const auto& heights = terrain.getHeightVector();
    const auto height = [&](int i, int j) { return heights[size_t(std::clamp(j, 0, n - 1)) * n + size_t(std::clamp(i, 0, n - 1))]; };
    const auto point = [&](int i, int j) {
      const double x = -half + i * cell, y = -half + j * cell;
      return Point{double(i), double(j), {rockMask(terrain, x, y), patchMask(x, y)}};
    };
    std::array<std::vector<::Vertex>, 3> vertices;
    std::array<std::vector<unsigned int>, 3> indices;
    const auto vertex = [&](int l, const Point& p) {
      const double x = -half + p.i * cell, y = -half + p.j * cell;
      const int i0 = std::min(int(p.i), n - 2), j0 = std::min(int(p.j), n - 2);
      const double fi = p.i - i0, fj = p.j - j0;
      const auto lerp = [&](auto f) { return (1 - fj) * ((1 - fi) * f(i0, j0) + fi * f(i0 + 1, j0)) + fj * ((1 - fi) * f(i0, j0 + 1) + fi * f(i0 + 1, j0 + 1)); };
      ::Vertex v;
      const double z = lerp(height);
      v.Position = glm::vec3(float(x), float(y), float(z));
      const double dx = lerp([&](int i, int j) { return (height(i + 1, j) - height(i - 1, j)) / (2 * cell); });
      const double dy = lerp([&](int i, int j) { return (height(i, j + 1) - height(i, j - 1)) / (2 * cell); });
      v.Normal = glm::normalize(glm::vec3(float(-dx), float(-dy), 1.f));
      v.TexCoords = glm::vec2(float(x / tiles_[l]), float(y / tiles_[l]));
      // Hollows (craters among them) are darker; the base ground varies in patches.
      const double around = 0.25 * (height(i0 - 2, j0) + height(i0 + 2, j0) + height(i0, j0 - 2) + height(i0, j0 + 2));
      const float hollow = float(std::clamp(1.0 + 1.2 * (z - around), 0.55, 1.08));
      const glm::vec3 color = l == 0 ? groundTint(x, y) : glm::vec3(float(0.94 + 0.06 * std::sin(0.21 * x + 0.5) * std::cos(0.17 * y)));
      v.Color = glm::vec4(hollow * color, 1.f);
      return v;
    };
    const auto emit = [&](int l, const Polygon& poly) {
      if (poly.size < 3) return;
      const auto base = unsigned(vertices[l].size());
      for (int k = 0; k < poly.size; ++k) vertices[l].push_back(vertex(l, poly.p[k]));
      for (int k = 1; k + 1 < poly.size; ++k) indices[l].insert(indices[l].end(), {base, base + unsigned(k), base + unsigned(k) + 1});
    };
    const auto split = [](const Polygon& poly, int m, Polygon& below, Polygon& above) {
      below.size = above.size = 0;
      for (int k = 0; k < poly.size; ++k) {
        const Point& a = poly.p[k];
        const Point& b = poly.p[(k + 1) % poly.size];
        Polygon& side = a.mask[m] > 0.5 ? above : below;
        side.p[side.size++] = a;
        if ((a.mask[m] > 0.5) != (b.mask[m] > 0.5)) {
          const double t = (0.5 - a.mask[m]) / (b.mask[m] - a.mask[m]);
          Point cut{a.i + t * (b.i - a.i), a.j + t * (b.j - a.j), {}};
          for (int q = 0; q < 2; ++q) cut.mask[q] = a.mask[q] + t * (b.mask[q] - a.mask[q]);
          below.p[below.size++] = cut;
          above.p[above.size++] = cut;
        }
      }
    };
    const auto triangle = [&](const Point& a, const Point& b, const Point& c) {
      Polygon all, rest, rock, base, patch;
      all.p[0] = a;
      all.p[1] = b;
      all.p[2] = c;
      all.size = 3;
      split(all, 0, rest, rock);
      emit(2, rock);
      split(rest, 1, base, patch);
      emit(0, base);
      emit(1, patch);
    };
    std::vector<Point> corners(size_t(per + 1) * size_t(per + 1));
    for (int j = 0; j <= per; ++j)
      for (int i = 0; i <= per; ++i) corners[size_t(j) * (per + 1) + size_t(i)] = point(cx * per + i, cy * per + j);
    const auto corner = [&](int i, int j) -> const Point& { return corners[size_t(j) * (per + 1) + size_t(i)]; };
    for (int j = 0; j < per; ++j)
      for (int i = 0; i < per; ++i) {
        const Point &a = corner(i, j), &b = corner(i + 1, j), &c = corner(i, j + 1), &d = corner(i + 1, j + 1);
        triangle(a, b, d);
        triangle(a, d, c);
      }
    for (int l = 0; l < 3; ++l) {
      if (indices[l].empty()) continue;
      auto meshes = std::make_shared<std::vector<std::shared_ptr<::OpenGLMesh>>>();
      meshes->push_back(std::make_shared<::OpenGLMesh>(std::move(vertices[l]), std::move(indices[l])));
      meshes->front()->material = materials_[l];
      const std::string name = "ground " + std::to_string(chunk) + " " + std::to_string(l) + " " + std::to_string(revision_++);
      viewer.addVisualCustomMesh(name, meshes, glm::vec4(1.f), false);
      chunkVisuals_[size_t(chunk)].push_back(name);
    }
  }

  Terrain kind_ = Terrain::Hills;
  int samples_ = 0;
  double extent_ = 0.0;
  std::array<raisin::Material, 3> materials_;
  std::array<double, 3> tiles_{};
  std::vector<std::vector<std::string>> chunkVisuals_;
  std::vector<int> dirty_;
  std::vector<std::string> outer_;
  size_t revision_ = 0;
};

// A body the game simulates as a primitive and draws with a model: the primitive is hidden and the model
// follows it.
struct Dressing {
  raisim::SingleBodyObject* body = nullptr;
  std::string visual;
  Vec3 offset = Vec3::Zero();
  std::shared_ptr<raisin::Visuals> mesh;
};

// Plants, trees, stones and the props around the arena, placed per terrain; trees, stones and barriers
// collide (their .rasset colliders).
class Scenery {
 public:
  struct Placement {
    double x, y, z, yaw, scale;
    int type;
  };

  void build(raisim::World& world, Viewer& viewer, const raisim::HeightMap& terrain, Terrain kind, const std::string& assets,
             int quality, std::mt19937& random, const std::vector<std::pair<Vec3, Vec3>>& buildings,
             const Vec3& start, std::vector<raisim::Object*>& colliders) {
    clear(viewer);
    quality_ = size_t(std::clamp(quality, 0, 3));
    const auto unit = [&]() { return uniform(random, 0.0, 1.0); };
    const double river = riverHalfWidth(kind);
    const auto blocked = [&](double x, double y, double margin) {
      if (std::hypot(x - start.x(), y - start.y()) < 14.0) return true;
      if (riverDistance(kind, x, y) < river + margin) return true;
      if (std::max(std::abs(x), std::abs(y)) > kArena - 3.0) return true;
      for (const auto& [center, half] : buildings)
        if (std::abs(x - center.x()) < half.x() + margin && std::abs(y - center.y()) < half.y() + margin) return true;
      return false;
    };
    const auto slope = [&](double x, double y) {
      const double dx = terrain.getHeight(x + 1, y) - terrain.getHeight(x - 1, y), dy = terrain.getHeight(x, y + 1) - terrain.getHeight(x, y - 1);
      return 0.5 * std::sqrt(dx * dx + dy * dy);
    };
    const std::string forest = assets + "/forest/", props = assets + "/tank_survival/props/";

    // Trees: pines and firs in groves.
    const int trees = kind == Terrain::Hills ? 110 : kind == Terrain::Ridges ? 150 : kind == Terrain::Fortress ? 35 : kind == Terrain::Town ? 14 : 0;
    std::vector<Placement> treePlacements;
    const auto grove = [](double x, double y) { return 0.5 + 0.5 * std::sin(x * 0.045 + 1.3) * std::cos(y * 0.05 - 0.4); };
    for (int attempts = 0; int(treePlacements.size()) < trees && attempts < trees * 80; ++attempts) {
      const double x = uniform(random, -kArena, kArena), y = uniform(random, -kArena, kArena);
      if (blocked(x, y, 4.0) || slope(x, y) > 0.55 || unit() > std::clamp(grove(x, y), 0.1, 1.0)) continue;
      bool crowded = false;
      for (const auto& p : treePlacements) crowded = crowded || std::hypot(p.x - x, p.y - y) < 3.0;
      if (crowded) continue;
      const double r = unit();
      const int type = kind == Terrain::Ridges || !kBroadleafTrees ? (r < 0.55 ? 0 : 1) : (r < 0.65 ? 2 : r < 0.85 ? 1 : 0);
      const double scale = (type == 2 ? 1.7 : 3.4) * (0.75 + 0.5 * unit());
      treePlacements.push_back({x, y, terrain.getHeight(x, y) - 0.05, unit() * 2 * kPi, scale, type});
    }
    const std::array<const char*, 3> treeNames{"pine_sapling_small", "fir_sapling", "tree_small_02"};
    for (int t = 0; t < (kBroadleafTrees ? 3 : 2); ++t) {
      const auto collider = raisim::Rasset::load(forest + treeNames[t] + "/model.rasset");
      std::vector<Placement> group;
      for (const auto& p : treePlacements)
        if (p.type == t) {
          group.push_back(p);
          for (auto* object : collider.addStaticColliders(world, {p.x, p.y, p.z}, p.yaw, p.scale, "", "wood")) colliders.push_back(object);
        }
      instanced(viewer, std::string("tree ") + treeNames[t], forest + treeNames[t] + "/model.gltf", group, true, 1.3f, -1, false);
    }

    // Woods on the hills outside the walls, which nothing reaches: drawn only.
    if (kind != Terrain::Dunes) {
      const int count = kind == Terrain::Town ? 120 : kind == Terrain::Fortress ? 260 : 700;
      std::array<std::vector<Placement>, 3> outside;
      for (int attempts = 0, placed = 0; placed < count && attempts < count * 20; ++attempts) {
        const double x = uniform(random, -520.0, 520.0), y = uniform(random, -520.0, 520.0);
        if (std::max(std::abs(x), std::abs(y)) < kArena + 8.0 || unit() > std::clamp(grove(x * 0.5, y * 0.5) * 1.3, 0.05, 1.0) ||
            riverDistance(kind, x, y) < river + 3.0)
          continue;
        const double r = unit();
        const int type = kind == Terrain::Ridges || !kBroadleafTrees ? (r < 0.5 ? 0 : 1) : (r < 0.65 ? 2 : r < 0.85 ? 1 : 0);
        const double z = std::max(std::abs(x), std::abs(y)) < 0.5 * kExtent ? terrain.getHeight(x, y) : outerHeight(kind, x, y);
        outside[size_t(type)].push_back({x, y, z - 0.1, unit() * 2 * kPi, (type == 2 ? 1.9 : 3.8) * (0.75 + 0.5 * unit()), type});
        ++placed;
      }
      const std::array<const char*, 3> names{"pine_sapling_small", "fir_sapling", "tree_small_02"};
      for (int t = 0; t < 3; ++t)
        instanced(viewer, std::string("far ") + names[t], forest + names[t] + "/model.gltf", outside[size_t(t)], true, 1.3f, -1, true);
    }

    // Stones and boulders.
    const bool desert = kind == Terrain::Dunes || kind == Terrain::Fortress;
    const int stones = kind == Terrain::Town ? 0 : 36;
    std::vector<Placement> stonePlacements;
    for (int attempts = 0; int(stonePlacements.size()) < stones && attempts < 4000; ++attempts) {
      const double x = uniform(random, -kArena, kArena), y = uniform(random, -kArena, kArena);
      if (blocked(x, y, 3.0)) continue;
      const int type = desert ? int(unit() * 2) : int(unit() * 6);
      const double scale = desert ? 0.9 + 1.4 * unit() : 0.7 + 1.6 * unit();
      stonePlacements.push_back({x, y, terrain.getHeight(x, y) - 0.15 * scale, unit() * 2 * kPi, scale, type});
    }
    for (int t = 0; t < (desert ? 2 : 6); ++t) {
      const std::string folder = desert ? props + (t == 0 ? "namaqualand_boulder_04" : "namaqualand_boulder_05")
                                        : forest + "rock_moss_" + std::to_string(t);
      const auto collider = raisim::Rasset::load(folder + "/model.rasset");
      std::vector<Placement> group;
      for (const auto& p : stonePlacements)
        if (p.type == t) {
          group.push_back(p);
          for (auto* object : collider.addStaticColliders(world, {p.x, p.y, p.z}, p.yaw, p.scale, "", "default")) colliders.push_back(object);
        }
      instanced(viewer, "stone " + std::to_string(t), folder + "/model.gltf", group, false, 0.0f, -1, true);
    }

    // Shrubs and stumps.
    std::vector<Placement> shrubs;
    const int shrubCount = kind == Terrain::Hills ? 90 : kind == Terrain::Dunes ? 70 : kind == Terrain::Fortress ? 40 : kind == Terrain::Town ? 25 : 30;
    for (int attempts = 0; int(shrubs.size()) < shrubCount && attempts < 5000; ++attempts) {
      const double x = uniform(random, -kArena, kArena), y = uniform(random, -kArena, kArena);
      if (blocked(x, y, 1.0) || slope(x, y) > 0.5) continue;
      shrubs.push_back({x, y, terrain.getHeight(x, y) - 0.05, unit() * 2 * kPi, 0.6 + 0.6 * unit(), desert ? 2 : int(unit() * 2)});
    }
    const std::array<const char*, 3> shrubNames{"shrub_02", "shrub_04", "wild_rooibos_bush"};
    for (int t = 0; t < 3; ++t) {
      std::vector<Placement> group;
      for (const auto& p : shrubs)
        if (p.type == t) group.push_back({p.x, p.y, p.z, p.yaw, t == 0 ? 0.35 * p.scale : t == 1 ? 3.5 * p.scale : p.scale, t});
      instanced(viewer, std::string("shrub ") + shrubNames[t], props + shrubNames[t] + "/model.gltf", group, true, 0.6f, -1, true);
    }
    if (kind == Terrain::Hills || kind == Terrain::Ridges) {
      const auto collider = raisim::Rasset::load(props + "tree_stump_01/model.rasset");
      std::vector<Placement> stumps;
      for (int attempts = 0; int(stumps.size()) < 18 && attempts < 2000; ++attempts) {
        const double x = uniform(random, -kArena, kArena), y = uniform(random, -kArena, kArena);
        if (blocked(x, y, 2.0)) continue;
        stumps.push_back({x, y, terrain.getHeight(x, y) - 0.05, unit() * 2 * kPi, 0.8 + 0.4 * unit(), 0});
        for (auto* object : collider.addStaticColliders(world, {x, y, stumps.back().z}, stumps.back().yaw, stumps.back().scale, "", "wood"))
          colliders.push_back(object);
      }
      instanced(viewer, "stumps", props + "tree_stump_01/model.gltf", stumps, false, 0.0f, -1, true);
    }

    // The arena's edge: concrete barriers along the walls the game builds.
    std::vector<Placement> barriers;
    for (int side = 0; side < 4; ++side)
      for (double s = -kArena + 1.0; s < kArena; s += 2.0) {
        const double x = side < 2 ? (side ? kArena : -kArena) : s, y = side < 2 ? s : (side == 3 ? kArena : -kArena);
        if (riverDistance(kind, x, y) < river + 1.5) continue;
        barriers.push_back({x, y, terrain.getHeight(x, y) - 0.05, side < 2 ? 0.5 * kPi : 0.0, 1.3, 0});
      }
    instanced(viewer, "barriers", props + "concrete_road_barrier/model.gltf", barriers, false, 0.0f, -1, false);

    // Grass in 32 m tiles, one batch per tile and plant, thinned out with distance (updateGrass()).
    const std::array<const char*, 5> grassNames{"grass_medium_01", "grass_medium_02", "grass_bermuda_01", "dandelion_01", "fern_02"};
    const double base = kind == Terrain::Hills ? 60000 : kind == Terrain::Fortress ? 14000 : kind == Terrain::Ridges ? 9000 : kind == Terrain::Town ? 3000 : 2500;
    const int count = int(base * kGrassDensity[size_t(std::clamp(quality, 0, 3))]);
    std::vector<Placement> grass;
    for (int attempts = 0; int(grass.size()) < count && attempts < count * 3; ++attempts) {
      const double x = uniform(random, -kArena, kArena), y = uniform(random, -kArena, kArena);
      if (std::max(std::abs(x), std::abs(y)) > kArena - 1.0 || slope(x, y) > 0.45 || riverDistance(kind, x, y) < river + 0.3) continue;
      bool inside = false;
      for (const auto& [center, half] : buildings)
        inside = inside || (std::abs(x - center.x()) < half.x() && std::abs(y - center.y()) < half.y());
      if (inside) continue;
      const double r = unit();
      int type = r < 0.45 ? 0 : r < 0.8 ? 1 : r < 0.9 ? 2 : r < 0.95 ? 3 : 4;
      if (kind != Terrain::Hills) type = r < 0.6 ? 2 : r < 0.9 ? 1 : 4;
      const double size = 1.1 * (type == 0 ? 2.4 : type == 1 ? 1.1 : type == 2 ? 1.6 : type == 3 ? 1.3 : 1.0);
      grass.push_back({x, y, terrain.getHeight(x, y) - 0.02, unit() * 2 * kPi, size * (0.7 + 0.6 * unit()), type});
    }
    constexpr int kTiles = 6;
    constexpr double kTileSize = 2.0 * kArena / kTiles;
    grassTiles_.assign(size_t(kTiles * kTiles), {});
    for (int tile = 0; tile < kTiles * kTiles; ++tile) {
      auto& entry = grassTiles_[size_t(tile)];
      entry.center = glm::vec2(float(-kArena + (tile % kTiles + 0.5) * kTileSize), float(-kArena + (tile / kTiles + 0.5) * kTileSize));
      for (int t = 0; t < int(grassNames.size()); ++t) {
        std::vector<Placement> group;
        for (const auto& p : grass)
          if (p.type == t && std::clamp(int((p.x + kArena) / kTileSize), 0, kTiles - 1) + kTiles * std::clamp(int((p.y + kArena) / kTileSize), 0, kTiles - 1) == tile)
            group.push_back(p);
        if (auto batch = instanced(viewer, std::string(grassNames[t]) + " " + std::to_string(tile), forest + grassNames[t] + "/model.gltf",
                                   group, true, 0.5f, tile, true))
          entry.batches.push_back(batch);
      }
    }
  }

  // Draws the grass at full density near the camera, thinner farther out, and none beyond its reach.
  void updateGrass(const glm::vec3& camera) {
    const float reach = kFoliageReach[quality_];
    for (auto& tile : grassTiles_) {
      const float distance = std::max(0.f, glm::length(tile.center - glm::vec2(camera)) - 22.f);
      for (auto& batch : tile.batches) {
        batch->setMaxRenderedInstances(distance > reach ? 1 : 0);
        batch->setRenderedInstanceStride(distance > 0.7f * reach ? 3 : distance > 0.35f * reach ? 2 : 1);
      }
    }
  }

  void clear(Viewer& viewer) {
    for (const auto& name : names_) viewer.removeInstancedVisuals(name);
    names_.clear();
    grassTiles_.clear();
  }

 private:
  struct GrassTile {
    glm::vec2 center;
    std::vector<std::shared_ptr<raisin::InstancedVisuals>> batches;
  };

  std::shared_ptr<raisin::InstancedVisuals> instanced(Viewer& viewer, const std::string& name, const std::string& mesh,
                                                      const std::vector<Placement>& placements, bool foliage, float sway,
                                                      int tile, bool tint) {
    if (placements.empty()) return nullptr;
    std::vector<raisin::InstancedVisuals::InstanceSpec> instances;
    instances.reserve(placements.size());
    for (const auto& p : placements) {
      raisin::InstancedVisuals::InstanceSpec instance;
      instance.position = {float(p.x), float(p.y), float(p.z)};
      instance.orientation = {float(std::cos(p.yaw * .5)), 0.f, 0.f, float(std::sin(p.yaw * .5))};
      instance.scale = glm::vec3(float(p.scale));
      instance.colorWeight = tint ? float(std::fmod(std::abs(p.x * 0.37 + p.y * 0.71), 1.0) * 0.5 + 0.25) : 0.f;
      instances.push_back(instance);
    }
    auto visual = viewer.addInstancedVisuals(name, raisim::Shape::Mesh, glm::vec3(1), glm::vec4(1),
                                             tint ? glm::vec4(.85f, .92f, .75f, 1) : glm::vec4(1), mesh, true);
    visual->addInstances(instances);
    visual->setAutomaticMeshLodEnabled(true);
    if (foliage) {
      // Trees and shrubs keep their leaves farther out; grass simplifies unseen at the renderer's
      // default, and the shadows keep theirs.
      if (tile < 0) {
        visual->setFoliageMeshLodPixelError(kFoliagePixelError[quality_]);
        visual->setFoliageShadowMeshLodPixelError(2.5f);
      }
      visual->setProjectedLodPolicy(true, kFoliageMinPixels[quality_], 8);
      visual->setShadowFoliageLodPolicy(true, 2.5f, 16);
      visual->setSortTransparentInstances(false);
      visual->configureFoliageWind(0, sway, sway > 1 ? .025f : .09f, .7f, .2f);
      visual->setFoliageImpostorPolicy(false);
    }
    if (tile >= 0) visual->setFoliageDistanceFade(0.75f * kFoliageReach[quality_], kFoliageReach[quality_], false);
    names_.push_back(name);
    return visual;
  }

  std::vector<std::string> names_;
  std::vector<GrassTile> grassTiles_;
  size_t quality_ = 2;
};

// Appends a box of the given size, centred at `center` and turned by `yaw` about z, with texture
// coordinates in metres over `tile`.
inline void appendBox(std::vector<::Vertex>& vertices, std::vector<unsigned int>& indices, const Vec3& size, const Vec3& center,
                      double yaw, double tile) {
  const Vec3 h = 0.5 * size;
  const double c = std::cos(yaw), s = std::sin(yaw);
  const auto turn = [&](const Vec3& v) { return Vec3(c * v.x() - s * v.y(), s * v.x() + c * v.y(), v.z()); };
  for (int axis = 0; axis < 3; ++axis)
    for (int sign = -1; sign <= 1; sign += 2) {
      const int u = (axis + 1) % 3, v = (axis + 2) % 3;
      Vec3 normal = Vec3::Zero();
      normal[axis] = sign;
      const Vec3 n = turn(normal);
      const auto base = unsigned(vertices.size());
      for (const auto& [a, b] : {std::array<int, 2>{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}) {
        Vec3 p = Vec3::Zero();
        p[axis] = sign * h[axis];
        p[u] = a * h[u];
        p[v] = b * h[v];
        const Vec3 w = center + turn(p);
        ::Vertex vertex;
        vertex.Position = glm::vec3(float(w.x()), float(w.y()), float(w.z()));
        vertex.Normal = glm::vec3(float(n.x()), float(n.y()), float(n.z()));
        const double along = axis == 2 ? p.x() : (axis == 0 ? p.y() : p.x()), up = axis == 2 ? p.y() : p.z();
        const double offset = axis == 2 ? 0.0 : center.x() * c + center.y() * s;
        vertex.TexCoords = glm::vec2(float((along + offset) / tile), float((up + (axis == 2 ? 0.0 : center.z())) / tile));
        vertices.push_back(vertex);
      }
      if (sign > 0) indices.insert(indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
      else indices.insert(indices.end(), {base, base + 2, base + 1, base, base + 3, base + 2});
    }
}

// A mesh of the given vertices drawn with `material`.
inline std::shared_ptr<raisin::Visuals> meshVisual(Viewer& viewer, const std::string& name, std::vector<::Vertex> vertices,
                                                   std::vector<unsigned int> indices, const raisin::Material& material) {
  auto meshes = std::make_shared<std::vector<std::shared_ptr<::OpenGLMesh>>>();
  meshes->push_back(std::make_shared<::OpenGLMesh>(std::move(vertices), std::move(indices)));
  meshes->front()->material = material;
  return viewer.addVisualCustomMesh(name, meshes, glm::vec4(1.f), false);
}

// A box drawn with a tiled PBR texture (walls of the fortress, rubble).
inline std::shared_ptr<raisin::Visuals> texturedBox(Viewer& viewer, const std::string& name, const Vec3& size,
                                                    const raisin::Material& material, double tile) {
  std::vector<::Vertex> vertices;
  std::vector<unsigned int> indices;
  appendBox(vertices, indices, size, Vec3::Zero(), 0.0, tile);
  return meshVisual(viewer, name, std::move(vertices), std::move(indices), material);
}

}  // namespace tank
