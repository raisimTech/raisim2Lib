// Captures a frame sequence from the rolling/spinning friction demo for the
// docs. The generator writes PNG frames into an intermediate directory;
// a follow-up CMake step encodes the frames into an MP4 video using ffmpeg.
//
// The scene mirrors the in-process example
// `examples/src/rayrai/dynamics/rayrai_rolling_spinning_friction.cpp` but
// runs entirely headless and uses a smaller grid so the video stays small.

#include "doc_image_common.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 400;
constexpr int kGrid = 6;             // 6x6 = 36 bodies, gives a clear pattern
// Long enough to show the full energy-decay arc: bodies start with linear
// and angular velocity, lose it to rolling/spinning friction, and finally
// drop into the sleeping state (which recolours them blue). 150 frames at
// 25 fps gives a 6 s loop; 40 sim steps per frame at dt=0.001 s gives
// 6 s of physics, which is enough for the slowest spheres to come to rest.
constexpr int kFrames = 150;
constexpr int kStepsPerFrame = 40;
constexpr double kTimestep = 0.001;

struct DemoBody {
  raisim::SingleBodyObject* object = nullptr;
  std::string activeAppearance;
  bool sleeping = false;
  bool sphereShape = false;
  int x = 0;
  int y = 0;
};

void resetBody(DemoBody& body, int grid) {
  const double spacing = 1.1;
  const double origin = -0.5 * spacing * static_cast<double>(grid - 1);
  const int x = body.x;
  const int y = body.y;

  if (body.sphereShape) {
    body.object->setPosition(origin + spacing * x, origin + spacing * y, 0.25);
    body.object->setOrientation(1.0, 0.0, 0.0, 0.0);
    body.object->setVelocity(1.7 + 0.08 * x, 0.4 + 0.05 * y, 0.0,
                             0.0, -11.0, 18.0);
  } else if ((x & 1) == 0) {
    body.object->setOrientation(0.7071067812, 0.0, 0.7071067812, 0.0);
    body.object->setPosition(origin + spacing * x, origin + spacing * y, 0.22);
    body.object->setVelocity(0.15 + 0.04 * x, 1.9 + 0.07 * y, 0.0,
                             14.0, 0.0, 0.0);
  } else {
    body.object->setOrientation(0.7071067812, -0.7071067812, 0.0, 0.0);
    body.object->setPosition(origin + spacing * x, origin + spacing * y, 0.22);
    body.object->setVelocity(1.9 + 0.07 * x, 0.15 + 0.04 * y, 0.0,
                             0.0, -14.0, 0.0);
  }
  body.sleeping = false;
  body.object->setAppearance(body.activeAppearance);
}

}  // namespace

int main(int argc, char** argv) {
  const auto outputDir = doc_image::resolveOutputDir(argc, argv);
  const char* framesDirEnv =
    std::getenv("RAISIM_DOC_ROLLING_FRICTION_FRAMES_DIR");
  const auto framesDir = (framesDirEnv && framesDirEnv[0] != '\0')
    ? std::filesystem::path(framesDirEnv)
    : std::filesystem::temp_directory_path() /
        "raisim_doc_rolling_friction_frames";
  std::filesystem::create_directories(framesDir);

  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_rolling_friction"))
    doc_image::finishAndExit(1);

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(kTimestep);
  world->setERP(0.0, 0.0);
  world->setContactSolverParam(1.0, 1.0, 1.0, 120, 1e-10);
  world->setGravity({0.0, 0.0, -9.81});
  world->setSleepingEnabled(true);
  world->setSleepingParameters(0.012, 0.03, 25);
  auto* ground = world->addGround(0.0, "ground");
  ground->setAppearance("checkerboard");
  // The same setMaterialPairProp overload introduced in v2.3.0: friction,
  // restitution, restitution threshold, static friction, static threshold
  // velocity, rolling friction, spinning friction.
  world->setMaterialPairProp("ground", "body",
                             1.0, 0.0, 0.0, 1.0, 1e-3, 0.12, 0.08);

  std::vector<DemoBody> bodies;
  bodies.reserve(static_cast<size_t>(kGrid * kGrid));
  for (int y = 0; y < kGrid; ++y) {
    for (int x = 0; x < kGrid; ++x) {
      DemoBody body;
      body.sphereShape = ((x + y) & 1) == 0;
      body.x = x; body.y = y;
      body.activeAppearance = body.sphereShape ?
        "0.95, 0.42, 0.12, 1.0" : "0.15, 0.70, 0.28, 1.0";
      if (body.sphereShape) {
        body.object = world->addSphere(0.25, 1.0, "body");
      } else {
        body.object = world->addCylinder(0.22, 0.55, 1.0, "body");
      }
      body.object->setAppearance(body.activeAppearance);
      bodies.push_back(body);
      resetBody(bodies.back(), kGrid);
    }
  }

  raisin::RayraiWindow renderer(world, kWidth, kHeight);
  const auto preset = raisin::RayraiWindow::RenderQualityPreset::High;
  auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(preset);
  quality.colorMode = raisin::ViewerColorMode::AcesApprox;
  quality.pbrToneMapping = true;
  doc_image::applyCommonSceneOptions(quality, preset);
  renderer.setRenderQualitySettings(quality);

  // Closer framing without narrowing the lens: keep the example's original
  // 65° horizontal FOV and just move the eye in toward the grid so
  // individual bodies are easy to read.
  doc_image::setCameraLookAt(renderer.getCamera(),
                             glm::vec3(2.7f, -3.8f, 2.3f),
                             glm::vec3(0.0f, 0.0f, 0.20f),
                             /*horizontalFovDeg=*/65.0f);

  raisin::RayraiWindow::RenderOverrides overrides;
  overrides.doShadows = true;

  for (int frame = 0; frame < kFrames; ++frame) {
    for (int i = 0; i < kStepsPerFrame; ++i) {
      world->integrate();
    }
    // Update sleeping appearance to recolour bodies that have come to rest.
    for (auto& body : bodies) {
      const bool sleeping = world->isObjectSleeping(body.object);
      if (sleeping != body.sleeping) {
        body.object->setAppearance(sleeping ? "0.20, 0.55, 1.00, 1.0"
                                            : body.activeAppearance);
        body.sleeping = sleeping;
      }
    }

    char name[64];
    std::snprintf(name, sizeof(name), "frame_%03d.png", frame);
    const auto framePath = framesDir / name;
    if (!doc_image::captureScene(renderer, kWidth, kHeight, framePath,
                                  /*supersampleScale=*/1, overrides)) {
      doc_image::finishAndExit(1);
    }
  }

  std::printf("doc_image: wrote %d frames into %s\n", kFrames,
              framesDir.string().c_str());
  doc_image::finishAndExit(0);
}
