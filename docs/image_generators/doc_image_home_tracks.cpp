// Records the home page's tracked-vehicle clip: the robot of the templated_tracked_robot example
// drives over a scatter of loose concrete blocks and climbs a three-step concrete stair with its
// flippers raised. The frames are encoded into rsc/docs/video/home/tile_tracks.mp4.
//
// Only CC0 assets and RaiSim's own robot: the asphalt and concrete textures and the sky are the
// city example's Poly Haven files (rsc/city/textures and rsc/city/sky, CC0, listed in
// rsc/city/ATTRIBUTION.md).

#include "doc_image_common.hpp"

#include <glbinding/gl/gl.h>
#include <rayrai/helper.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Rendered at twice the 960 x 600 of the home tile, which the encoder scales down.
constexpr int kWidth = 1920, kHeight = 1200;
constexpr int kFps = 30;
constexpr double kSeconds = 7.5;
constexpr int kFrames = int(kSeconds * kFps);
constexpr double kTimeStep = 0.001;
constexpr float kPi = 3.14159265359f;

/// the example's robot with the example's parameters and controller: sprockets at 5 rad/s,
/// flippers raised 0.6 rad (as in doc_image_tracks.cpp)
raisim::ArticulatedSystem* addTrackedRobot(raisim::World& world, const std::filesystem::path& rsc) {
  const std::vector<raisim::World::ParameterContainer> params = {
      {"rho", "500"}, {"base_length", "1.0"}, {"base_width", "0.15"}, {"wheel_y_offset", "0.14"},
      {"track_y_offset", "0.24"}, {"wheel_width", "0.08"}, {"wheel_radius", "0.17775"},
      {"flipper_radius", "0.16"}, {"track_thickness", "0.015"}};
  auto* robot = world.addArticulatedSystem((rsc / "templatedTrackedRobot/trackedTemplate.urdf").string(), params);
  robot->setBasePos({-0.4, 0., 0.25});
  Eigen::VectorXd kp = Eigen::VectorXd::Zero(robot->getDOF()), kd = kp, velocity = kp;
  Eigen::VectorXd position = Eigen::VectorXd::Zero(robot->getGeneralizedCoordinateDim());
  for (const char* sprocket : {"left_sprocket", "right_sprocket"}) {
    kd[robot->getGeneralizedVelocityIndex(sprocket)] = 50.;
    velocity[robot->getGeneralizedVelocityIndex(sprocket)] = 5.;
  }
  for (const char* flipper : {"left_flipper_joint", "right_flipper_joint"}) {
    kp[robot->getGeneralizedVelocityIndex(flipper)] = 250.;
    kd[robot->getGeneralizedVelocityIndex(flipper)] = 50.;
    position[robot->getJoint(flipper).getIdxInGeneralizedCoordinate()] = -0.6;
  }
  robot->setPdGains(kp, kd);
  robot->setPdTarget(position, velocity);
  return robot;
}

/// a Poly Haven texture set: diffuse, OpenGL normal and AO-roughness-metal maps
raisin::Material polyHavenMaterial(const std::filesystem::path& folder, const std::string& name, const char* resolution,
                          const glm::vec4& tint) {
  auto material = raisin::Material::pbr(name, tint, 0.f, 0.9f);
  const auto file = (folder / name / name).string();
  material.albedoMap = raisin::RayraiWindow::loadTextureWithTiling((file + "_diff_" + resolution + ".jpg").c_str(), true);
  material.normalMap = raisin::RayraiWindow::loadTextureWithTiling((file + "_nor_gl_" + resolution + ".jpg").c_str(), false);
  material.ormMap = raisin::RayraiWindow::loadTextureWithTiling((file + "_arm_" + resolution + ".jpg").c_str(), false);
  return material;
}

raisin::MeshList singleMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices,
                            const raisin::Material& material) {
  auto meshes = std::make_shared<std::vector<std::shared_ptr<OpenGLMesh>>>();
  meshes->push_back(std::make_shared<OpenGLMesh>(std::move(vertices), std::move(indices)));
  meshes->front()->material = material;
  return meshes;
}

/// a box centered at the origin whose texture repeats every `tile` metres on each face
raisin::MeshList texturedBox(const glm::vec3& size, float tile, const raisin::Material& material) {
  std::vector<Vertex> vertices;
  std::vector<unsigned int> indices;
  const glm::vec3 half = 0.5f * size;
  for (int axis = 0; axis < 3; ++axis)
    for (const float sign : {1.f, -1.f}) {
      // u and v span the face, with u x v along the axis.
      const int u = (axis + 1) % 3, v = (axis + 2) % 3;
      const auto base = unsigned(vertices.size());
      const float corners[4][2] = {{-1.f, -1.f}, {1.f, -1.f}, {1.f, 1.f}, {-1.f, 1.f}};
      for (const auto& corner : corners) {
        Vertex vertex;
        vertex.Position[axis] = sign * half[axis];
        vertex.Position[u] = corner[0] * half[u];
        vertex.Position[v] = corner[1] * half[v];
        vertex.Normal = glm::vec3(0.f);
        vertex.Normal[axis] = sign;
        vertex.TexCoords = glm::vec2(vertex.Position[u], vertex.Position[v]) / tile;
        vertices.push_back(vertex);
      }
      if (sign > 0.f) indices.insert(indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
      else indices.insert(indices.end(), {base, base + 2, base + 1, base, base + 3, base + 2});
    }
  return singleMesh(std::move(vertices), std::move(indices), material);
}

/// a square ground plane at z = 0, in `cells` x `cells` quads, its texture repeating every `tile` metres
raisin::MeshList texturedPlane(float size, int cells, float tile, const raisin::Material& material) {
  std::vector<Vertex> vertices;
  std::vector<unsigned int> indices;
  for (int j = 0; j <= cells; ++j)
    for (int i = 0; i <= cells; ++i) {
      Vertex vertex;
      vertex.Position = glm::vec3(size * (float(i) / cells - 0.5f), size * (float(j) / cells - 0.5f), 0.f);
      vertex.TexCoords = glm::vec2(vertex.Position) / tile;
      vertices.push_back(vertex);
    }
  for (int j = 0; j < cells; ++j)
    for (int i = 0; i < cells; ++i) {
      const unsigned a = unsigned(j * (cells + 1) + i), b = a + 1, c = a + unsigned(cells + 1), d = c + 1;
      indices.insert(indices.end(), {a, b, d, a, d, c});
    }
  return singleMesh(std::move(vertices), std::move(indices), material);
}

/// Direction of the sun in the sky HDR, from the sun towards the scene (as in rayrai_offroad_buggy).
const glm::vec3 kSunDirection = glm::normalize(glm::vec3(.5556f, .3738f, -.7427f));

/// the lighting of the rayrai_offroad_buggy example: a clear weather sky lights the scene, Poly
/// Haven's partly cloudy HDR sky (CC0, rsc/city/sky) is the background, and a bright sun casts
/// cascaded shadows
void configureRendering(raisin::RayraiWindow& viewer, const std::filesystem::path& sky) {
  using Viewer = raisin::RayraiWindow;
  auto quality = Viewer::defaultRenderQualitySettings(Viewer::RenderQualityPreset::High);
  quality.gamma = 2.2f;
  quality.colorMode = raisin::ViewerColorMode::AcesApprox;
  quality.pbrExposure = 0.75f;
  quality.pbrEnvironmentLightingTint = glm::vec3(.65f);
  quality.depthOfFieldEnabled = false;
  quality.reflectiveGround = false;
  quality.viewerMsaaSamples = 4;
  quality.directionalShadowCascadeCount = 3;
  quality.directionalShadowCascadeMaxDistance = 40;
  quality.addViewerFillLights = false;
  quality.highFidelityPbr = true;
  viewer.setRenderQualitySettings(quality);
  Viewer::WeatherSettings weather;
  weather.enabled = true;
  weather.preset = Viewer::WeatherPreset::Clear;
  weather.useExplicitSunAngles = true;
  weather.sunAzimuthDegrees = 214;  // as in the HDR
  weather.sunElevationDegrees = 48;
  weather.sunDiskSize = .001f;
  weather.cloudCoverage = .12f;
  weather.visibilityMeters = 6000.f;
  weather.fogColor = glm::vec3(.66f, .72f, .78f);
  viewer.setWeatherSettings(weather);
  viewer.generateWeatherSkyEnvironment(256, 32, false);
  viewer.setSceneEnvironmentBackground(viewer.acquireSceneHdrCubemapCached(sky.string(), 1024), 1.f);
  auto& sun = viewer.getLight();
  sun.type = raisin::LightType::DIRECTIONAL;
  sun.setShadowsEnabled(true);
  sun.setShadowResolution(4096);
  sun.setShadowParams(.0008f, 1.f, 1.25f);
  viewer.setShadowOrtho(12, .1f, 60);
  sun.direction = kSunDirection;
  sun.ambient = glm::vec3(.13f);
  sun.diffuse = glm::vec3(1, .95f, .86f) * 5.f;
  sun.specular = glm::vec3(5);
}

struct Block {
  raisim::Box* body;
  std::shared_ptr<raisin::Visuals> visual;
};

void record(const std::filesystem::path& frames, const std::filesystem::path& rsc) {
  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(kTimeStep);
  world->addGround(0., "ground")->setAppearance("hidden");

  // A stair of three 0.2 m steps with 0.8 m treads, then a 3 m landing.
  struct Slab { double x, length, height; };
  std::vector<Slab> slabs;
  for (int i = 0; i < 3; ++i) {
    const double height = 0.2 * (i + 1);
    slabs.push_back(i < 2 ? Slab{3.0 + 0.8 * i, 0.8, height} : Slab{3.0 + 0.8 * i + 1.1, 3.0, height});
  }
  for (const auto& slab : slabs) {
    auto* box = world->addBox(slab.length, 2.2, slab.height, 100.);
    box->setPosition(slab.x, 0., 0.5 * slab.height);
    box->setBodyType(raisim::BodyType::STATIC);
    box->setAppearance("hidden");
  }
  // Loose blocks in the robot's path, which its tracks push aside and drive over.
  std::vector<Block> blocks;
  for (int i = 0; i < 6; ++i) {
    auto* block = world->addBox(0.22, 0.22, 0.11, 4.);
    block->setPosition(1.3 + 0.27 * (i % 3), -0.3 + 0.3 * (i / 3) + 0.05 * (i % 2), 0.06);
    block->setOrientation(std::cos(0.15 * i), 0., 0., std::sin(0.15 * i));
    block->setAppearance("hidden");
    blocks.push_back({block, nullptr});
  }
  auto* robot = addTrackedRobot(*world, rsc);

  raisin::RayraiWindow viewer(world, kWidth, kHeight);
  configureRendering(viewer, rsc / "city/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr");

  const auto textures = rsc / "city/textures";
  // The ground: asphalt under the course, drawn as one textured plane.
  const auto asphalt = polyHavenMaterial(textures, "asphalt_02", "2k", glm::vec4(0.9f, 0.9f, 0.9f, 1.f));
  viewer.addVisualCustomMesh("course ground", texturedPlane(400.f, 80, 5.f, asphalt), glm::vec4(1.f), false);
  const auto stepConcrete = polyHavenMaterial(textures, "concrete_floor_01", "1k", glm::vec4(0.9f, 0.9f, 0.9f, 1.f));
  const auto blockConcrete = polyHavenMaterial(textures, "concrete_floor_01", "1k", glm::vec4(0.6f, 0.6f, 0.6f, 1.f));
  int index = 0;
  for (const auto& slab : slabs) {
    auto visual = viewer.addVisualCustomMesh("step " + std::to_string(index++),
        texturedBox({float(slab.length), 2.2f, float(slab.height)}, 1.2f, stepConcrete), glm::vec4(1.f), false);
    visual->setPosition(slab.x, 0., 0.5 * slab.height);
  }
  index = 0;
  for (auto& block : blocks) {
    block.visual = viewer.addVisualCustomMesh("block " + std::to_string(index++),
        texturedBox({0.22f, 0.22f, 0.11f}, 0.5f, blockConcrete), glm::vec4(1.f), false);
  }
  const auto syncBlocks = [&] {
    for (auto& block : blocks) {
      const auto p = block.body->getPosition();
      const auto q = block.body->getQuaternion();
      block.visual->setPosition(p[0], p[1], p[2]);
      block.visual->setOrientation(q[0], q[1], q[2], q[3]);
    }
  };

  // The camera follows the robot on its right, swinging from behind it to its front as it climbs.
  // Frames come from the viewer's own render, as on screen: captures through an external camera
  // leave the shadows off the textured meshes of the course.
  auto& camera = viewer.getCamera();
  glm::vec3 target(0.f);
  const auto placeCamera = [&](int frame, bool snap) {
    const auto base = robot->getBasePosition();
    const glm::vec3 wanted(float(base[0]) + 0.35f, 0.f, std::max(float(base[2]), 0.3f));
    target = snap ? wanted : target + 0.12f * (wanted - target);
    const float s = float(frame) / float(kFrames - 1);
    const float heading = kPi * (-115.f + 40.f * s) / 180.f;
    const float distance = 2.6f - 0.15f * s;
    doc_image::setCameraLookAt(camera,
                               target + glm::vec3(distance * std::cos(heading), distance * std::sin(heading), 0.6f),
                               target, 46.f);
    camera.farPlane = camera.zFar = 400.f;
  };

  const auto render = [&] { viewer.update(kWidth, kHeight, /*save=*/false, 0, 0, /*headless=*/true); };
  syncBlocks();
  placeCamera(0, true);
  viewer.setAsyncMeshLoadingEnabled(false);
  for (int i = 0; i < 5; ++i) render();  // Settle temporal effects at the first pose.

  const char* previewFrames = std::getenv("RAISIM_DOC_GALLERY_FRAME_COUNT");
  const int frameCount = previewFrames ? std::clamp(std::atoi(previewFrames), 1, kFrames) : kFrames;
  const int totalSteps = int(std::lround(kSeconds / kTimeStep));
  std::vector<unsigned char> rgba(size_t(kWidth) * kHeight * 4);
  for (int frame = 0; frame < frameCount; ++frame) {
    syncBlocks();
    placeCamera(frame, frame == 0);
    render();
    gl::glBindTexture(gl::GL_TEXTURE_2D, camera.getFinalTexture());
    gl::glGetTexImage(gl::GL_TEXTURE_2D, 0, gl::GL_RGBA, gl::GL_UNSIGNED_BYTE, rgba.data());
    gl::glBindTexture(gl::GL_TEXTURE_2D, 0);
    char name[32];
    std::snprintf(name, sizeof(name), "frame_%03d.png", frame);
    if (!doc_image::writePng(frames / name, kWidth, kHeight, rgba, /*flipVertical=*/true))
      throw std::runtime_error("could not write " + std::string(name));
    // Real time: 1/30 s of simulation per frame.
    const int steps = (frame + 1) * totalSteps / kFrames - frame * totalSteps / kFrames;
    for (int i = 0; i < steps; ++i) world->integrate();
    if (!robot->getGeneralizedCoordinate().e().allFinite()) throw std::runtime_error("non-finite robot state");
    if ((frame + 1) % 15 == 0 || frame + 1 == frameCount) {
      const auto base = robot->getBasePosition();
      std::printf("doc_image: home_tracks frame %d/%d, robot at x %.2f m, z %.2f m\n", frame + 1, frameCount,
                  base[0], base[2]);
      std::fflush(stdout);
    }
  }
  // The clip shows the climb: the robot must end on the landing, 0.6 m up.
  if (frameCount == kFrames && robot->getBasePosition()[2] < 0.7)
    throw std::runtime_error("the robot did not climb the stair");
}

}  // namespace

int main(int argc, char** argv) {
  const auto output = doc_image::resolveOutputDir(argc, argv);
  const auto rsc = argc >= 3 ? std::filesystem::path(argv[2])
                             : std::filesystem::weakly_canonical(output / ".." / ".." / "..");
  const char* override = std::getenv("RAISIM_DOC_GALLERY_FRAMES_DIR");
  const auto frames = override ? std::filesystem::path(override) : output / "home_tracks_frames";
  std::filesystem::create_directories(frames);
  if (std::filesystem::exists(rsc / "activation.raisim"))
    raisim::World::setActivationKey((rsc / "activation.raisim").string());
  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_home_tracks")) doc_image::finishAndExit(1);
  try {
    record(frames, rsc);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "doc_image: %s\n", e.what());
    doc_image::finishAndExit(1);
  }
  doc_image::finishAndExit(0);
}
