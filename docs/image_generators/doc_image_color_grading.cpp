// Generates rayrai_grade_<preset>.png for Neutral / Warm / Cool / Cinematic /
// Bleach. The only setting that varies between images is the
// `viewerColorGradePreset` — everything else is whatever the High preset
// ships with. Library defaults must produce a good base image; tuning
// belongs in rayrai, not here.
//
// Used by docs/sections/Rayrai.md "Tone mapping, exposure, and color grading".

#include "doc_image_common.hpp"

namespace {

constexpr int kWidth = 960;
constexpr int kHeight = 540;

struct GradeSpec {
  const char* fileSuffix;
  raisin::ColorGradePreset preset;
};

const GradeSpec kGrades[] = {
    {"neutral",   raisin::ColorGradePreset::Neutral},
    {"warm",      raisin::ColorGradePreset::Warm},
    {"cool",      raisin::ColorGradePreset::Cool},
    {"cinematic", raisin::ColorGradePreset::Cinematic},
    {"bleach",    raisin::ColorGradePreset::Bleach},
};

}  // namespace

int main(int argc, char** argv) {
  const auto outputDir = doc_image::resolveOutputDir(argc, argv);
  doc_image::OffscreenContext gl;
  if (!gl.init("doc_image_color_grading")) doc_image::finishAndExit(1);

  for (const auto& spec : kGrades) {
    auto world = std::make_shared<raisim::World>();
    raisin::RayraiWindow renderer(world, kWidth, kHeight);

    const auto preset = raisin::RayraiWindow::RenderQualityPreset::High;
    auto quality = raisin::RayraiWindow::defaultRenderQualitySettings(preset);
    // The only variables across the five images.
    quality.viewerColorGradePreset = spec.preset;
    quality.viewerColorGradeStrength = 1.0f;
    doc_image::applyCommonSceneOptions(quality, preset);
    renderer.setRenderQualitySettings(quality);

    doc_image::buildStarterScene(*world, renderer);

    const auto path = outputDir / (std::string("rayrai_grade_") + spec.fileSuffix + ".png");
    if (!doc_image::captureScene(renderer, kWidth, kHeight, path))
      doc_image::finishAndExit(1);
    std::printf("doc_image: wrote %s\n", path.string().c_str());
  }
  doc_image::finishAndExit(0);
}
