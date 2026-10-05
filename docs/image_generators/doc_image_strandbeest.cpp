// Records the walking Strandbeest for the documentation and README galleries.
// PNG frames are composited into a GIF by CMake using ffmpeg.

#include "doc_image_common.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace {

constexpr int kWidth = 960;
constexpr int kHeight = 600;
constexpr double kTimeStep = 0.002;
constexpr int kStepsPerFrame = 20;  // 25 frames per second of simulation
constexpr int kFrames = 100;

}  // namespace

int main(int argc, char** argv) {
  const auto outputDir = doc_image::resolveOutputDir(argc, argv);
  const auto rsc = argc >= 3 ? std::filesystem::path(argv[2])
                             : std::filesystem::weakly_canonical(outputDir / ".." / ".." / "..");
  const char* framesOverride = std::getenv("RAISIM_DOC_STRANDBEEST_FRAMES_DIR");
  const auto framesDir = framesOverride ? std::filesystem::path(framesOverride)
                                       : outputDir / "strandbeest_frames";
  std::filesystem::create_directories(framesDir);
  if (std::filesystem::exists(rsc / "activation.raisim"))
    raisim::World::setActivationKey((rsc / "activation.raisim").string());

  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_strandbeest")) doc_image::finishAndExit(1);

  auto world = std::make_shared<raisim::World>();
  world->setTimeStep(kTimeStep);
  world->setSleepingEnabled(false);
  world->addGround(0., "ground")->setAppearance("checkerboard");
  auto* beest = world->addArticulatedSystem((rsc / "strandbeest/strandbeest.urdf").string());

  // Use the same single-crank velocity controller as strandbeest_closed_loops.
  const size_t crank = beest->getGeneralizedVelocityIndex("joint_crossbar_crank");
  Eigen::VectorXd gain = Eigen::VectorXd::Zero(beest->getDOF()), target = gain;
  gain[crank] = 200.;
  target[crank] = 2. * std::acos(-1.);
  beest->setPdGains(Eigen::VectorXd::Zero(beest->getDOF()), gain);
  beest->setPdTarget(beest->getGeneralizedCoordinate().e(), target);
  for (int step = 0; step < 650; ++step) world->integrate();

  raisin::RayraiWindow renderer(world, kWidth, kHeight);
  const auto preset = raisin::RayraiWindow::RenderQualityPreset::High;
  auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(preset);
  quality.colorMode = raisin::ViewerColorMode::AcesApprox;
  quality.pbrToneMapping = true;
  doc_image::applyCommonSceneOptions(quality, preset);
  renderer.setRenderQualitySettings(quality);
  raisin::RayraiWindow::RenderOverrides overrides;
  overrides.doShadows = true;

  for (int frame = 0; frame < kFrames; ++frame) {
    const auto& gc = beest->getGeneralizedCoordinate();
    if (!gc.e().allFinite()) {
      std::fprintf(stderr, "doc_image: Strandbeest state is not finite at frame %d\n", frame);
      doc_image::finishAndExit(1);
    }
    const glm::vec3 center(float(gc[0]), float(gc[1]), 0.45f);
    doc_image::setCameraLookAt(renderer.getCamera(), center + glm::vec3(1.6f, -2.0f, 1.0f), center, 50.f);
    // The first capture loads scene assets, bakes the sky, and warms the renderer.
    if (frame == 0 && !doc_image::captureScene(renderer, kWidth, kHeight,
                                               outputDir / "constraints_strandbeest.png", 2, overrides))
      doc_image::finishAndExit(1);

    char name[64];
    std::snprintf(name, sizeof(name), "frame_%03d.png", frame);
    renderer.update(kWidth, kHeight, /*save=*/false, /*saveW=*/0, /*saveH=*/0, /*headless=*/true);
    auto capture = renderer.captureSupersampledRgba(renderer.getCamera(), 2, overrides);
    if (capture.rgba.empty() || !doc_image::writePng(framesDir / name, capture.width, capture.height,
                                                   std::move(capture.rgba), /*flipVertical=*/false))
      doc_image::finishAndExit(1);
    for (int step = 0; step < kStepsPerFrame; ++step) world->integrate();
    if ((frame + 1) % 25 == 0) {
      std::printf("doc_image: Strandbeest frame %d/%d\n", frame + 1, kFrames);
      std::fflush(stdout);
    }
  }

  std::printf("doc_image: wrote %d Strandbeest frames into %s\n", kFrames, framesDir.string().c_str());
  doc_image::finishAndExit(0);
}
